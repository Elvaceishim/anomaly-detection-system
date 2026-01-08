# Security Policy

This document outlines the security boundaries for LLM/MCP access to the anomaly detection system.

## Never Exposed to MCP or LLM

The following data is **never** exposed through MCP tools:

| Category | Examples | Reason |
|----------|----------|--------|
| **Raw PII** | Full card numbers, names, addresses, SSNs | Privacy regulations (GDPR, PCI-DSS) |
| **Training Labels** | `is_fraud` column from training data | Prevents label leakage |
| **Fraud Outcomes** | Whether past transactions were confirmed fraud | Prevents outcome leakage |
| **Full User Histories** | Complete transaction logs per user | Privacy; limits blast radius |
| **Internal Thresholds** | Exact decision boundaries, cutoff values | Prevents gaming the system |
| **Model Weights** | LightGBM tree structures, coefficients | Intellectual property; attack surface |
| **Raw Database Access** | Direct SQL queries, table scans | Least-privilege violation |

## What MCP Tools CAN Access

| Tool | Access Level | Data Returned |
|------|--------------|---------------|
| `get_transaction_summary` | Read-only | Amount, merchant category, timestamp, location (coarse), risk score |
| `get_user_behavior_snapshot` | Read-only | Rolling stats (count, avg amount), velocity metrics, behavioral flags |
| `get_anomaly_signals` | Read-only | Percentile ranks, deviation scores, spike indicators |
| `get_model_explanation` | Read-only | Top contributing features (names only), relative importance |
| `log_human_decision` | Write (audit) | Stores: txn_id, decision, notes, timestamp, analyst_id |

## Design Principles

1. **Least Privilege**: LLM gets minimum data needed for its job (explanation, not decision-making)
2. **Aggregation Over Raw**: Provide summaries, not raw records
3. **No Write Access**: Except structured audit logging
4. **Explicit Boundaries**: Every tool documents what it does NOT return
5. **Audit Everything**: All MCP calls are logged for review

## Reporting Security Issues

If you discover a security vulnerability, please open a private issue on GitHub or contact the maintainer directly.
