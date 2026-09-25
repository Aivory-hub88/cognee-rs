use chrono::{DateTime, Utc};
use cognee_models::{Data, Dataset};
use cognee_utils::tracing_keys::{COGNEE_DB_ROW_COUNT, COGNEE_DB_SYSTEM};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait,
    IntoActiveModel, PaginatorTrait, QueryFilter, sea_query::Expr,
};
use tracing::{Span, instrument};
use uuid::Uuid;

use crate::conversions::map_sea_err;
use crate::database_system_label;
use crate::entities::{data, dataset, dataset_data};
use crate::types::DatabaseError;
use crate::uuid_hex;

#[instrument(
    name = "cognee.db.relational.data.create_data",
    level = "info",
    skip_all,
    fields(cognee.db.system = tracing::field::Empty),
    err,
)]
pub async fn create_data(db: &DatabaseConnection, d: Data) -> Result<Data, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    data::ActiveModel::from(&d)
        .insert(db)
        .await
        .map_err(map_sea_err)?;
    Ok(d)
}

#[instrument(
    name = "cognee.db.relational.data.get_data",
    level = "info",
    skip_all,
    fields(
        cognee.db.system = tracing::field::Empty,
        cognee.db.row_count = tracing::field::Empty,
    ),
    err,
)]
pub async fn get_data(db: &DatabaseConnection, id: Uuid) -> Result<Option<Data>, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    let result = data::Entity::find_by_id(uuid_hex::to_hex(id))
        .one(db)
        .await
        .map_err(map_sea_err)
        .map(|opt| opt.map(Data::from))?;
    Span::current().record(
        COGNEE_DB_ROW_COUNT,
        if result.is_some() { 1i64 } else { 0i64 },
    );
    Ok(result)
}

#[instrument(
    name = "cognee.db.relational.data.delete_data",
    level = "info",
    skip_all,
    fields(cognee.db.system = tracing::field::Empty),
    err,
)]
pub async fn delete_data(db: &DatabaseConnection, id: Uuid) -> Result<(), DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    data::Entity::delete_by_id(uuid_hex::to_hex(id))
        .exec(db)
        .await
        .map_err(map_sea_err)?;
    Ok(())
}

#[instrument(
    name = "cognee.db.relational.data.update_data",
    level = "info",
    skip_all,
    fields(cognee.db.system = tracing::field::Empty),
    err,
)]
pub async fn update_data(db: &DatabaseConnection, d: Data) -> Result<Data, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    let mut model = data::ActiveModel::from(&d);
    model.updated_at = Set(Some(Utc::now()));
    model.update(db).await.map_err(map_sea_err)?;
    Ok(d)
}

#[instrument(
    name = "cognee.db.relational.data.count_data_dataset_links",
    level = "info",
    skip_all,
    fields(
        cognee.db.system = tracing::field::Empty,
        cognee.db.row_count = tracing::field::Empty,
    ),
    err,
)]
pub async fn count_data_dataset_links(
    db: &DatabaseConnection,
    data_id: Uuid,
) -> Result<usize, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    let count: u64 = dataset_data::Entity::find()
        .filter(dataset_data::Column::DataId.eq(uuid_hex::to_hex(data_id)))
        .count(db)
        .await
        .map_err(map_sea_err)?;
    Span::current().record(COGNEE_DB_ROW_COUNT, count as i64);
    Ok(count as usize)
}

/// Update only the `token_count` column for a Data record.
///
/// Mirrors the Python `update_document_token_count()` in
/// `cognee/tasks/documents/extract_chunks_from_documents.py`.
#[instrument(
    name = "cognee.db.relational.data.update_data_token_count",
    level = "info",
    skip_all,
    fields(cognee.db.system = tracing::field::Empty),
    err,
)]
pub async fn update_data_token_count(
    db: &DatabaseConnection,
    data_id: Uuid,
    token_count: i64,
) -> Result<(), DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    // Single UPDATE instead of find-then-update (2 round-trips -> 1). The
    // rows-affected count preserves the previous NotFound behaviour.
    let result = data::Entity::update_many()
        .col_expr(data::Column::TokenCount, Expr::value(token_count))
        .col_expr(data::Column::UpdatedAt, Expr::value(Some(Utc::now())))
        .filter(data::Column::Id.eq(uuid_hex::to_hex(data_id)))
        .exec(db)
        .await
        .map_err(map_sea_err)?;

    if result.rows_affected == 0 {
        return Err(DatabaseError::NotFound(format!("Data {data_id} not found")));
    }
    Ok(())
}

/// Update `last_accessed` for a batch of Data records identified by their IDs.
///
/// This is a no-op when `data_ids` is empty.
#[instrument(
    name = "cognee.db.relational.data.update_last_accessed",
    level = "info",
    skip_all,
    fields(cognee.db.system = tracing::field::Empty),
    err,
)]
pub async fn update_last_accessed(
    db: &DatabaseConnection,
    data_ids: &[Uuid],
    timestamp: DateTime<Utc>,
) -> Result<(), DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    if data_ids.is_empty() {
        return Ok(());
    }

    // Single UPDATE ... WHERE id IN (...) instead of N×(find + update) round-trips.
    let hex_ids: Vec<_> = data_ids.iter().map(|id| uuid_hex::to_hex(*id)).collect();
    data::Entity::update_many()
        .col_expr(data::Column::LastAccessed, Expr::value(Some(timestamp)))
        .filter(data::Column::Id.is_in(hex_ids))
        .exec(db)
        .await
        .map_err(map_sea_err)?;

    Ok(())
}

/// Clear `pipeline_status` JSON entries keyed by the given `dataset_id`
/// from all `Data` records linked to that dataset via the `dataset_data`
/// junction table.
///
/// This mirrors the Python cleanup in `delete_dataset.py` lines 33-54.
/// Must be called **before** the junction rows are removed (before
/// `detach_data_from_dataset` or `delete_dataset`), since the junction is
/// needed to find related `Data` records.
///
/// Returns the number of `Data` records whose `pipeline_status` was modified.
#[instrument(
    name = "cognee.db.relational.data.clear_pipeline_status_for_dataset",
    level = "info",
    skip_all,
    fields(
        cognee.db.system = tracing::field::Empty,
        cognee.db.row_count = tracing::field::Empty,
    ),
    err,
)]
pub async fn clear_pipeline_status_for_dataset(
    db: &DatabaseConnection,
    dataset_id: Uuid,
) -> Result<usize, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    // Find all data IDs linked to this dataset via the junction table
    let junction_rows = dataset_data::Entity::find()
        .filter(dataset_data::Column::DatasetId.eq(uuid_hex::to_hex(dataset_id)))
        .all(db)
        .await
        .map_err(map_sea_err)?;

    let data_ids: Vec<String> = junction_rows.into_iter().map(|j| j.data_id).collect();
    if data_ids.is_empty() {
        Span::current().record(COGNEE_DB_ROW_COUNT, 0i64);
        return Ok(0);
    }

    let dataset_id_str = uuid_hex::to_hex(dataset_id);
    let mut updated_count = 0usize;

    // Read the linked Data rows in chunks instead of one find per id (N reads).
    // Chunking keeps each `IN (...)` under the driver's bound-variable cap
    // (SQLite ~32766, Postgres 65535) — mirroring `PROVENANCE_INSERT_BATCH` in
    // graph_storage — so a dataset with more items than the cap can't overflow.
    // Updates stay per-row because each row's pipeline_status JSON is mutated
    // independently, and only rows that actually change are written back.
    const READ_CHUNK: usize = 500;
    let mut models = Vec::with_capacity(data_ids.len());
    for chunk in data_ids.chunks(READ_CHUNK) {
        models.extend(
            data::Entity::find()
                .filter(data::Column::Id.is_in(chunk.to_vec()))
                .all(db)
                .await
                .map_err(map_sea_err)?,
        );
    }

    for model in models {
        let Some(ref status_json) = model.pipeline_status else {
            continue;
        };

        let mut parsed: serde_json::Value = serde_json::from_str(status_json)
            .unwrap_or(serde_json::Value::Object(Default::default()));

        let serde_json::Value::Object(ref mut top_map) = parsed else {
            continue;
        };

        let mut modified = false;
        for (_pipeline_name, inner) in top_map.iter_mut() {
            if let serde_json::Value::Object(inner_map) = inner
                && inner_map.remove(&dataset_id_str).is_some()
            {
                modified = true;
            }
        }

        if !modified {
            continue;
        }

        // Remove pipeline entries whose inner map is now empty
        top_map.retain(|_, v| !matches!(v, serde_json::Value::Object(m) if m.is_empty()));

        let new_status = if top_map.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&parsed).map_err(|e| {
                DatabaseError::QueryError(format!("Failed to serialize pipeline_status: {e}"))
            })?)
        };

        let mut active = model.into_active_model();
        active.pipeline_status = Set(new_status);
        active.updated_at = Set(Some(Utc::now()));
        active.update(db).await.map_err(map_sea_err)?;
        updated_count += 1;
    }

    Span::current().record(COGNEE_DB_ROW_COUNT, updated_count as i64);
    Ok(updated_count)
}

/// Per-data-item status value Python cognee writes under
/// `pipeline_status["cognify_pipeline"][<dataset_id>]` once an item has been
/// cognified (`run_tasks_data_item`), and what incremental cognify skips on.
pub const COGNIFY_DATA_ITEM_COMPLETED: &str = "DATA_ITEM_PROCESSING_COMPLETED";

/// True when a Data row's `pipeline_status` JSON marks it cognified for
/// `dataset_id`. Malformed or missing JSON reads as "not yet", so a bad row
/// is re-processed rather than silently dropped from the graph.
pub fn is_cognify_completed_for_dataset(pipeline_status: Option<&str>, dataset_id: Uuid) -> bool {
    let Some(raw) = pipeline_status else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) else {
        return false;
    };
    parsed
        .get("cognify_pipeline")
        .and_then(|inner| inner.get(uuid_hex::to_hex(dataset_id)))
        .and_then(|v| v.as_str())
        == Some(COGNIFY_DATA_ITEM_COMPLETED)
}

/// Record that `data_ids` were cognified for `dataset_id`: sets
/// `pipeline_status["cognify_pipeline"][<dataset_id>]` to
/// [`COGNIFY_DATA_ITEM_COMPLETED`], preserving every other entry. The inverse
/// of [`clear_cognify_pipeline_status_for_data`], which delete already calls,
/// so forgetting an item makes the next cognify pick it up again.
#[instrument(
    name = "cognee.db.relational.data.mark_cognify_completed_for_data",
    level = "info",
    skip_all,
    fields(
        cognee.db.system = tracing::field::Empty,
        cognee.db.row_count = tracing::field::Empty,
    ),
    err,
)]
pub async fn mark_cognify_completed_for_data(
    db: &DatabaseConnection,
    data_ids: &[Uuid],
    dataset_id: Uuid,
) -> Result<u64, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    let dataset_key = uuid_hex::to_hex(dataset_id);
    let mut updated: u64 = 0;
    for data_id in data_ids {
        let Some(model) = data::Entity::find_by_id(uuid_hex::to_hex(*data_id))
            .one(db)
            .await
            .map_err(map_sea_err)?
        else {
            continue;
        };
        let mut top: serde_json::Map<String, serde_json::Value> = model
            .pipeline_status
            .as_deref()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| match v {
                serde_json::Value::Object(m) => Some(m),
                _ => None,
            })
            .unwrap_or_default();
        let mut inner = match top.remove("cognify_pipeline") {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        inner.insert(
            dataset_key.clone(),
            serde_json::Value::String(COGNIFY_DATA_ITEM_COMPLETED.into()),
        );
        top.insert("cognify_pipeline".into(), serde_json::Value::Object(inner));
        let serialized = serde_json::to_string(&serde_json::Value::Object(top)).map_err(|e| {
            DatabaseError::QueryError(format!("Failed to serialize pipeline_status: {e}"))
        })?;
        let mut active = model.into_active_model();
        active.pipeline_status = Set(Some(serialized));
        active.updated_at = Set(Some(Utc::now()));
        active.update(db).await.map_err(map_sea_err)?;
        updated += 1;
    }
    Span::current().record(COGNEE_DB_ROW_COUNT, updated as i64);
    Ok(updated)
}

/// Clear only the `cognify_pipeline` entry for `dataset_id` from a single
/// Data record's `pipeline_status` JSON. All other entries are preserved.
///
/// Mirrors Python `_forget_data_memory` lines 343-348.
#[instrument(
    name = "cognee.db.relational.data.clear_cognify_pipeline_status_for_data",
    level = "info",
    skip_all,
    fields(
        cognee.db.system = tracing::field::Empty,
    ),
    err,
)]
pub async fn clear_cognify_pipeline_status_for_data(
    db: &DatabaseConnection,
    data_id: Uuid,
    dataset_id: Uuid,
) -> Result<(), DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    let model = data::Entity::find_by_id(uuid_hex::to_hex(data_id))
        .one(db)
        .await
        .map_err(map_sea_err)?;

    let Some(model) = model else {
        return Ok(());
    };

    let Some(ref status_json) = model.pipeline_status else {
        return Ok(());
    };

    let mut parsed: serde_json::Value =
        serde_json::from_str(status_json).unwrap_or(serde_json::Value::Object(Default::default()));

    let serde_json::Value::Object(ref mut top_map) = parsed else {
        return Ok(());
    };

    let dataset_id_str = uuid_hex::to_hex(dataset_id);
    let Some(inner) = top_map.get_mut("cognify_pipeline") else {
        return Ok(());
    };
    let modified = if let serde_json::Value::Object(inner_map) = inner {
        inner_map.remove(&dataset_id_str).is_some()
    } else {
        false
    };

    if !modified {
        return Ok(());
    }

    // Remove `cognify_pipeline` if its inner map is now empty.
    top_map.retain(|k, v| {
        k != "cognify_pipeline" || !matches!(v, serde_json::Value::Object(m) if m.is_empty())
    });

    let new_status = if top_map.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&parsed).map_err(|e| {
            DatabaseError::QueryError(format!("Failed to serialize pipeline_status: {e}"))
        })?)
    };

    let mut active = model.into_active_model();
    active.pipeline_status = Set(new_status);
    active.updated_at = Set(Some(Utc::now()));
    active.update(db).await.map_err(map_sea_err)?;
    Ok(())
}

#[instrument(
    name = "cognee.db.relational.data.list_datasets_for_data",
    level = "info",
    skip_all,
    fields(
        cognee.db.system = tracing::field::Empty,
        cognee.db.row_count = tracing::field::Empty,
    ),
    err,
)]
pub async fn list_datasets_for_data(
    db: &DatabaseConnection,
    data_id: Uuid,
) -> Result<Vec<Dataset>, DatabaseError> {
    Span::current().record(COGNEE_DB_SYSTEM, database_system_label(db));
    let pairs = data::Entity::find_by_id(uuid_hex::to_hex(data_id))
        .find_with_related(dataset::Entity)
        .all(db)
        .await
        .map_err(map_sea_err)?;
    let datasets: Vec<Dataset> = pairs
        .into_iter()
        .flat_map(|(_, ds_list)| ds_list)
        .map(Dataset::from)
        .collect();
    Span::current().record(COGNEE_DB_ROW_COUNT, datasets.len() as i64);
    Ok(datasets)
}

#[cfg(test)]
mod cognify_status_tests {
    use super::*;

    #[test]
    fn reads_per_dataset_cognify_completion() {
        let ds = Uuid::parse_str("45534e3f-01ce-5b52-9c2b-5568436c5417").unwrap_or_default();
        let other = Uuid::new_v4();
        let done = format!(
            r#"{{"cognify_pipeline":{{"{}":"DATA_ITEM_PROCESSING_COMPLETED"}},"add_pipeline":{{"x":"y"}}}}"#,
            uuid_hex::to_hex(ds)
        );
        assert!(is_cognify_completed_for_dataset(Some(&done), ds));
        assert!(!is_cognify_completed_for_dataset(Some(&done), other));
        assert!(!is_cognify_completed_for_dataset(None, ds));
        assert!(!is_cognify_completed_for_dataset(Some("not json"), ds));
        assert!(!is_cognify_completed_for_dataset(Some(r#"{"cognify_pipeline":"weird"}"#), ds));
    }
}
