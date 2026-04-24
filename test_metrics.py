import pandas as pd
import pytest

from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
            "age": [30, 40, 28, 35, 32, 45],
            "monthly_income": [4000, 6000, 5000, 7000, 4500, 8000],
            "job_satisfaction": [2, 4, 2, 4, 3, 3],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "travel_frequency": ["Frequent", "Rarely", "Frequent", "Rarely", "Occasional", "Rarely"],
            "years_at_company": [2, 8, 1, 6, 3, 12],
            "attrition": ["Yes", "No", "Yes", "No", "No", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_basic(sample_df):
    assert attrition_rate(sample_df) == 33.33


def test_attrition_rate_all_leave(sample_df):
    all_leave = sample_df.copy()
    all_leave["attrition"] = "Yes"
    assert attrition_rate(all_leave) == 100.0


def test_attrition_rate_none_leave(sample_df):
    none_leave = sample_df.copy()
    none_leave["attrition"] = "No"
    assert attrition_rate(none_leave) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_counts(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0


def test_attrition_by_department_zero_attrition(sample_df):
    result = attrition_by_department(sample_df)
    hr = result[result["department"] == "HR"].iloc[0]
    assert hr["leavers"] == 0
    assert hr["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_yes_group(sample_df):
    result = attrition_by_overtime(sample_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    assert yes_row["employees"] == 3
    assert yes_row["leavers"] == 2
    assert yes_row["attrition_rate"] == 66.67


def test_attrition_by_overtime_no_group(sample_df):
    result = attrition_by_overtime(sample_df)
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert no_row["employees"] == 3
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_leavers(sample_df):
    result = average_income_by_attrition(sample_df)
    leavers_income = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    assert leavers_income == 4500.0  # (4000 + 5000) / 2


def test_average_income_stayers(sample_df):
    result = average_income_by_attrition(sample_df)
    stayers_income = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert stayers_income == 6375.0  # (6000 + 7000 + 4500 + 8000) / 4


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)


def test_satisfaction_summary_rate_uses_group_denominator(sample_df):
    # satisfaction 2: 2 employees, both left → 100%, not 2/2*100 from total leavers
    result = satisfaction_summary(sample_df)
    row = result[result["job_satisfaction"] == 2].iloc[0]
    assert row["total_employees"] == 2
    assert row["leavers"] == 2
    assert row["attrition_rate"] == 100.0


def test_satisfaction_summary_zero_attrition_group(sample_df):
    result = satisfaction_summary(sample_df)
    row = result[result["job_satisfaction"] == 4].iloc[0]
    assert row["leavers"] == 0
    assert row["attrition_rate"] == 0.0
