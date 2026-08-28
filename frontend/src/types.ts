export type Decision = 'approve' | 'review' | 'block'

export interface BookingRequest {
  booking_id: string
  agent_id: string
  account_age_days: number
  total_bookings_90d: number
  chargebacks_90d: number
  cancellations_90d: number
  bookings_24h: number
  recent_holds_24h: number
  recent_late_cancellations_90d: number
  seats_requested: number
  hours_until_departure: number
  ip_country: string
  card_country: string
  payment_attempts_10m: number
  device_linked_to_fraud: boolean
  card_on_blocklist: boolean
  interaction_duration_seconds: number
  fields_pasted: number
  pointer_events: number
}

export interface RiskReason {
  code: string
  label: string
  contribution: number
  source: 'rule' | 'telemetry' | 'xgboost_shap'
}

export interface ComponentScore {
  score: number
  reasons: RiskReason[]
}

export interface AssessmentResult {
  booking_id: string
  decision: Decision
  overall_score: number
  payment_fraud: ComponentScore
  inventory_abuse: ComponentScore
  bot_likelihood: ComponentScore
  engine_mode: string
  policy_version: string
  summary: string
}

export interface AnalystBrief {
  provider: string
  text: string
  data_disclosure: string
}

export interface HealthStatus {
  status: string
  engine_mode: string
}

export interface MetricSet {
  threshold: number
  precision: number
  recall: number
  pr_auc: number
  roc_auc: number
  false_positive_rate: number
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
}

export interface ModelMetadata {
  generated_at_utc: string
  data_disclosure: string
  row_count: number
  split_strategy: string
  split_counts: { train: number; validation: number; test: number }
  test_metrics: {
    payment_fraud: MetricSet
    inventory_abuse: MetricSet
  }
}

export interface GraphNode {
  id: string
  label: string
  kind: 'agent' | 'device' | 'ip' | 'payment'
  status: string
}

export interface GraphData {
  provider: string
  data_disclosure: string
  nodes: GraphNode[]
  edges: { source: string; target: string; relationship: string }[]
}
