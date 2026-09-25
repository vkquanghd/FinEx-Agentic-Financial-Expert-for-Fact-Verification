# Quy ước đánh số migration — chốt 2026-08-04 (DSA v6 mục 2.3)

## Xung đột và cách giải quyết

Master FINAL (69tr, Phần I mục 13) đặt tên roadmap target là `012_source_registry` →
`025_compatibility`. Nhưng repo đã dùng **012/013 cho prediction store + feedback HITL**
(bản mini theo DSA v5 §4/§6) và cả hai ĐÃ CHẠY LIVE trên `thesis_postgres`.

**Chốt:**

1. `012_prediction_store.sql` và `013_feedback_hitl.sql` GIỮ NGUYÊN — migration đã áp
   dụng là bất biến, không đổi tên/sửa nội dung.
2. KHÔNG sửa gì trên database hiện tại — xung đột chỉ là đánh số file, không phải schema.
3. Migration tương lai theo roadmap Master FINAL đánh số từ **014**, theo bảng ánh xạ dưới.
4. Trong thesis (chương Future Work) dùng TÊN NỘI DUNG (source_registry, excel_ingest…),
   không dùng số, để không lệch giữa các tài liệu.

## Bảng ánh xạ số cũ (Master FINAL) → số mới (repo)

| Master FINAL | Repo (khi tạo) | Nội dung |
|---|---|---|
| 012_source_registry | **014**_source_registry | source systems/records/objects/observations/runs/artifacts |
| 013_excel_ingest | **015**_excel_ingest | workbook/version/sheet/period/line/candidate/cell |
| 014_financial_vintages | **016**_financial_vintages | taxonomy/concept/period/fact/status/views |
| 015_document_versions | **017**_document_versions | version/extraction/diff/chunks_v2 |
| 016_entity_security | **018**_entity_security | entity/security/identifier/alias |
| 017_events_news | **019**_events_news | events/links/actions/news |
| 018_quality | **020**_quality | rules/runs/results |
| 019_hitl | **021**_hitl_full | queue/decisions/attestations (mở rộng từ 013-mini) |
| 020_ml_data | **022**_ml_data | dataset/member/feature/split snapshots |
| 021_model_registry | **023**_model_registry_full | runs/versions/calibration/promotion/monitoring (mở rộng từ model_registry_lite trong 012) |
| 022_predictions | **024**_predictions_full | forecast/revision/RAG payload (mở rộng từ ml_prediction_outputs trong 012) |
| 023_lineage_answers | **025**_lineage_answers | lineage/answer/evidence/impact (evidence JSONB trong 013 unnest ra bảng thật) |
| 024_adjusted_market | **026**_adjusted_market | action ledger/adjusted series |
| 025_compatibility | **027**_compatibility | legacy views/backfill/deprecation |

Ghi chú nâng cấp: 021/023/024/025 (số mới) được thiết kế để MỞ RỘNG các bảng mini hiện
có (012/013) bằng ADD/bảng mới + view, KHÔNG phá schema đang chạy — đúng nguyên tắc
compatibility-first, không big-bang rewrite (Master FINAL Phần I mục 13).
