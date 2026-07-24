"""Catalog validation and exact/perceptual cross-pool overlap auditing."""

from __future__ import annotations

from collections import Counter

from microq_phase1.core.config import PoolConfig, ProjectConfig
from microq_phase1.core.errors import PoolPolicyError
from microq_phase1.core.logging import get_logger
from microq_phase1.application.overlap import CrossPoolOverlapDetector
from microq_phase1.domain.models import ImageRecord
from microq_phase1.domain.policy import StrictTwoPoolPolicy
from microq_phase1.infrastructure.catalogs import JsonlImageCatalog
from microq_phase1.infrastructure.hashing import ImageHasher, ImageSignature
from microq_phase1.infrastructure.jsonl import write_json_atomic

LOGGER = get_logger(__name__)


class PoolAuditService:
    """Enforce role, provenance, and image-overlap constraints before generation."""

    def __init__(
        self,
        policy: StrictTwoPoolPolicy,
        hasher: ImageHasher,
        overlap: CrossPoolOverlapDetector,
    ) -> None:
        self._policy = policy
        self._hasher = hasher
        self._overlap = overlap

    def audit(self, config: ProjectConfig) -> dict[str, object]:
        self._policy.validate_pool_configs(config.pools)
        records = self._load_records(config.pools)
        signatures = self._create_signatures(records)
        exact = self._overlap.exact(signatures, records)
        near = self._overlap.near(
            signatures,
            records,
            config.validation.perceptual_hamming_threshold,
        )
        license_violations = self._license_violations(
            records,
            config.validation.license_allowlist,
        )
        failed = bool(
            exact
            or license_violations
            or (near and config.validation.fail_on_near_duplicate)
        )
        report = {
            "status": "failed" if failed else "passed",
            "policy": "strict_two_pool",
            "pool_summary": self._summarize(records),
            "exact_cross_role_duplicates": exact,
            "near_cross_role_duplicates": near,
            "license_violations": license_violations,
            "perceptual_hamming_threshold": config.validation.perceptual_hamming_threshold,
        }
        write_json_atomic(config.validation.overlap_report_path, report)
        LOGGER.info(
            "pool_audit_completed",
            status=report["status"],
            exact_duplicates=len(exact),
            near_duplicates=len(near),
            license_violations=len(license_violations),
            report=str(config.validation.overlap_report_path),
        )
        if report["status"] == "failed":
            raise PoolPolicyError(
                "Strict pool audit failed. Inspect the overlap report before generation."
            )
        return report

    def _load_records(
        self,
        pools: list[PoolConfig],
    ) -> dict[str, tuple[PoolConfig, list[ImageRecord]]]:
        loaded: dict[str, tuple[PoolConfig, list[ImageRecord]]] = {}
        global_ids: set[tuple[str, str]] = set()
        for pool in pools:
            items = list(JsonlImageCatalog(pool.catalog_path).records())
            self._policy.validate_records(pool, items)
            for item in items:
                identity = (pool.name, item.image_id)
                if identity in global_ids:
                    raise PoolPolicyError(f"Duplicate image_id inside pool: {identity}")
                global_ids.add(identity)
            loaded[pool.name] = (pool, items)
        return loaded

    def _create_signatures(
        self,
        pools: dict[str, tuple[PoolConfig, list[ImageRecord]]],
    ) -> dict[str, list[ImageSignature]]:
        output: dict[str, list[ImageSignature]] = {}
        for pool_name, (_, records) in pools.items():
            signatures: list[ImageSignature] = []
            for index, record in enumerate(records, start=1):
                signatures.append(
                    self._hasher.signature(record.image_id, pool_name, record.image_path)
                )
                if index % 500 == 0:
                    LOGGER.info(
                        "pool_hashing_progress",
                        pool=pool_name,
                        completed=index,
                        total=len(records),
                    )
            output[pool_name] = signatures
        return output

    @staticmethod
    def _license_violations(
        pools: dict[str, tuple[PoolConfig, list[ImageRecord]]],
        allowlist: set[str],
    ) -> list[dict[str, str]]:
        violations: list[dict[str, str]] = []
        for pool_name, (_, records) in pools.items():
            for record in records:
                if record.license in allowlist:
                    continue
                violations.append(
                    {
                        "pool": pool_name,
                        "image_id": record.image_id,
                        "license": record.license,
                    }
                )
        return violations

    @staticmethod
    def _summarize(
        pools: dict[str, tuple[PoolConfig, list[ImageRecord]]],
    ) -> dict[str, object]:
        summary: dict[str, object] = {}
        for name, (pool, records) in pools.items():
            summary[name] = {
                "role": pool.role,
                "official_benchmark": pool.official_benchmark,
                "images": len(records),
                "modalities": dict(Counter(record.modality for record in records)),
                "source_datasets": len({record.source_dataset for record in records}),
            }
        return summary
