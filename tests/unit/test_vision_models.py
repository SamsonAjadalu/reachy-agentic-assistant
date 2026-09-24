"""ORM invariants for visual spatial memory."""

from __future__ import annotations

from sqlalchemy import JSON, String

from database.models.vision import (
    CameraCalibration,
    CameraSource,
    EvaluationDataset,
    EvaluationResult,
    EvaluationSample,
    EvidenceAsset,
    ModelRun,
    ModelVersion,
    ObjectDetection,
    ObjectEntity,
    ObjectEntityObservation,
    ObjectIdentityRevision,
    ObjectTrack,
    ObservationRequest,
    SceneSnapshot,
    VisualDeletionReceipt,
    VisualEmbedding,
    VisualEvent,
    VisualObservation,
    VisualSpatialRelation,
    VisualWatch,
    VisualZone,
)
from vision.enums import FORBIDDEN_METRIC_COLUMNS, WORLD_FRAME_STATUS_VALUE
from vision.repositories import assert_no_metric_columns

VISION_MODELS = (
    CameraSource,
    CameraCalibration,
    VisualZone,
    ModelVersion,
    ModelRun,
    EvaluationDataset,
    EvidenceAsset,
    ObservationRequest,
    SceneSnapshot,
    VisualObservation,
    ObjectTrack,
    ObjectEntity,
    ObjectDetection,
    ObjectEntityObservation,
    ObjectIdentityRevision,
    VisualSpatialRelation,
    VisualWatch,
    VisualEvent,
    VisualEmbedding,
    EvaluationSample,
    EvaluationResult,
    VisualDeletionReceipt,
)


class TestVisionSchemaInvariants:
    def test_all_expected_tables_are_registered(self) -> None:
        names = {model.__tablename__ for model in VISION_MODELS}
        assert len(names) == 22

    def test_every_vision_table_has_no_metric_columns(self) -> None:
        for model in VISION_MODELS:
            assert_no_metric_columns(model)
            names = {column.name for column in model.__table__.columns}
            assert not (names & FORBIDDEN_METRIC_COLUMNS)

    def test_no_sqlalchemy_json_columns(self) -> None:
        for model in VISION_MODELS:
            for column in model.__table__.columns:
                assert not isinstance(column.type, JSON), f"{model.__tablename__}.{column.name}"

    def test_world_frame_status_defaults_to_unknown(self) -> None:
        column = VisualObservation.__table__.c.world_frame_status
        assert column.default is not None
        assert column.default.arg == WORLD_FRAME_STATUS_VALUE

    def test_detections_cascade_from_observations(self) -> None:
        fks = list(ObjectDetection.__table__.c.observation_id.foreign_keys)
        assert any(fk.ondelete == "CASCADE" for fk in fks)

    def test_evidence_foreign_keys_set_null(self) -> None:
        for column_name in ("full_frame_evidence_id", "thumbnail_evidence_id"):
            fks = list(VisualObservation.__table__.c[column_name].foreign_keys)
            assert fks
            assert all(fk.ondelete == "SET NULL" for fk in fks)

    def test_evaluation_evidence_is_restrict(self) -> None:
        fks = list(EvaluationSample.__table__.c.evidence_id.foreign_keys)
        assert any(fk.ondelete == "RESTRICT" for fk in fks)

    def test_model_runs_restrict_model_versions(self) -> None:
        fks = list(ModelRun.__table__.c.model_version_id.foreign_keys)
        assert any(fk.ondelete == "RESTRICT" for fk in fks)

    def test_content_hash_is_unique_64(self) -> None:
        column = EvidenceAsset.__table__.c.content_hash
        assert isinstance(column.type, String)
        assert column.type.length == 64
        unique_names = {uq.name for uq in EvidenceAsset.__table__.constraints if uq.name}
        assert "uq_evidence_assets_content_hash" in unique_names

    def test_provenance_columns_have_no_foreign_keys(self) -> None:
        assert not VisualObservation.__table__.c.model_run_id.foreign_keys
        assert not ObjectEntity.__table__.c.last_seen_observation_id.foreign_keys
        assert isinstance(VisualObservation.__table__.c.model_run_id.type, String)
        assert VisualObservation.__table__.c.model_run_id.type.length == 36

    def test_person_flags_block_entities_and_embeddings(self) -> None:
        assert "is_person" in ObjectDetection.__table__.c
        assert "blocks_embedding" in ObjectDetection.__table__.c
        assert "blocks_entity" in ObjectDetection.__table__.c
        assert "is_person" in ObjectEntity.__table__.c
        assert "blocks_embeddings" in ObjectEntity.__table__.c

    def test_named_multi_column_indexes_are_explicit(self) -> None:
        names = {index.name for index in VisualObservation.__table__.indexes}
        assert "ix_visual_observations_zone_captured_at" in names
        assert "ix_visual_observations_trigger_captured_at" in names

    def test_idempotency_keys_exist_on_write_tables(self) -> None:
        for model in (
            VisualObservation,
            ObservationRequest,
            SceneSnapshot,
            VisualWatch,
            EvidenceAsset,
            VisualEvent,
            EvaluationResult,
        ):
            assert "idempotency_key" in model.__table__.c
