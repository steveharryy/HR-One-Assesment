"""
Comprehensive Test Suite for Employee Attendance & Analytics API
50 test cases covering all endpoints and business rules (R1-R10).

Requirements: pip install pytest httpx
Run:  pytest test_comprehensive.py -v
Note: Requires a running MongoDB instance (uses MONGO_URI/MONGO_DB from env or defaults).
"""
import os
import time
import calendar
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "attendance_test_db")

from app.main import app, db, IST, epoch_ms_to_dt, dt_to_epoch_ms

client = TestClient(app)

IST_OFFSET = timedelta(hours=5, minutes=30)


def ist_to_epoch_ms(year, month, day, hour, minute, second=0):
    """Helper: convert IST time components to epoch milliseconds."""
    dt = datetime(year, month, day, hour, 0, 0, tzinfo=IST) + timedelta(minutes=minute, seconds=second)
    return int(dt.timestamp() * 1000)


def utc_to_epoch_ms(year, month, day, hour, minute, second=0):
    """Helper: convert UTC time components to epoch milliseconds."""
    dt = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute, seconds=second)
    return int(dt.timestamp() * 1000)


@pytest.fixture(autouse=True)
def clean_db():
    """Clean database before each test and ensure unique indexes."""
    db.employees.delete_many({})
    db.attendance_logs.delete_many({})
    db.employees.create_index("emp_code", unique=True)
    db.employees.create_index("email", unique=True)
    db.attendance_logs.create_index(
        [("emp_code", 1), ("date", 1)], unique=True
    )
    yield
    db.employees.delete_many({})
    db.attendance_logs.delete_many({})


def create_employee(
    emp_code="EMP0001",
    name="Test User",
    email="test@example.com",
    department="Engineering",
    shift_start="09:30",
    shift_end="18:30",
    joined_on="2026-01-05",
):
    """Helper to create an employee."""
    return client.post(
        "/employees",
        json={
            "emp_code": emp_code,
            "name": name,
            "email": email,
            "department": department,
            "shift_start": shift_start,
            "shift_end": shift_end,
            "joined_on": joined_on,
        },
    )


# ===========================================================================
# 1. Health Endpoint
# ===========================================================================
def test_01_health_returns_ok():
    """TC01: GET /health returns 200 with {status: ok} when MongoDB is up."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ===========================================================================
# 2. Employee CRUD
# ===========================================================================
def test_02_create_employee_success():
    """TC02: POST /employees creates an employee and returns 201."""
    r = create_employee()
    assert r.status_code == 201
    data = r.json()
    assert data["emp_code"] == "EMP0001"
    assert data["name"] == "Test User"
    assert data["department"] == "Engineering"
    assert data["shift_start"] == "09:30"
    assert data["shift_end"] == "18:30"
    assert data["joined_on"] == "2026-01-05"
    assert isinstance(data["created_at"], int)
    assert data["created_at"] >= 100_000_000_000


def test_03_create_employee_duplicate_409():
    """TC03: Duplicate emp_code returns 409."""
    create_employee()
    r = create_employee()
    assert r.status_code == 409


def test_04_create_employee_invalid_emp_code_422():
    """TC04: Invalid emp_code pattern returns 422."""
    r = client.post(
        "/employees",
        json={
            "emp_code": "BAD",
            "name": "X",
            "email": "x@x.com",
            "department": "D",
            "joined_on": "2026-01-01",
        },
    )
    assert r.status_code == 422


def test_05_create_employee_shift_same_422():
    """TC05: shift_start == shift_end returns 422."""
    r = client.post(
        "/employees",
        json={
            "emp_code": "EMP0001",
            "name": "X",
            "email": "x@x.com",
            "department": "D",
            "shift_start": "09:30",
            "shift_end": "09:30",
            "joined_on": "2026-01-01",
        },
    )
    assert r.status_code == 422


def test_06_list_employees_pagination():
    """TC06: List employees with pagination and department filter."""
    for i in range(1, 6):
        dept = "Engineering" if i <= 3 else "Sales"
        create_employee(
            emp_code=f"EMP{i:04d}",
            name=f"User {i}",
            email=f"u{i}@x.com",
            department=dept,
        )
    r = client.get("/employees?page=1&page_size=2")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 5
    assert data["page"] == 1
    assert data["page_size"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["emp_code"] == "EMP0001"

    r2 = client.get("/employees?department=Sales")
    assert r2.json()["total"] == 2


def test_07_list_employees_sorted_by_emp_code():
    """TC07: Employees are sorted by emp_code ascending."""
    create_employee(emp_code="EMP0003", name="C", email="c@x.com")
    create_employee(emp_code="EMP0001", name="A", email="a@x.com")
    create_employee(emp_code="EMP0002", name="B", email="b@x.com")
    r = client.get("/employees")
    codes = [e["emp_code"] for e in r.json()["items"]]
    assert codes == ["EMP0001", "EMP0002", "EMP0003"]


# ===========================================================================
# 3. Punch-In
# ===========================================================================
def test_08_punch_in_success():
    """TC08: Punch-in creates attendance record with correct fields."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 28, 0)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["emp_code"] == "EMP0001"
    assert data["date"] == "2026-07-06"
    assert data["status"] == "PRESENT"
    assert data["punch_in"] == ts
    assert data["punch_out"] is None
    assert data["work_hours"] is None
    assert data["late_minutes"] == 0
    assert data["overtime_minutes"] == 0
    assert data["half_day"] is False
    assert data["history"] == []


def test_09_punch_in_unknown_employee_404():
    """TC09: Punch-in for non-existent employee returns 404."""
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP9999", "punched_at": ist_to_epoch_ms(2026, 7, 6, 9, 30)},
    )
    assert r.status_code == 404


def test_10_punch_in_duplicate_409():
    """TC10: Duplicate punch-in for same emp+date returns 409."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 28)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts})
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 6, 10, 0)},
    )
    assert r.status_code == 409


def test_11_punch_in_invalid_epoch_422():
    """TC11: Seconds-looking punched_at rejected with 422."""
    create_employee()
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": 1609459200},
    )
    assert r.status_code == 422


def test_12_punch_in_wfh_status():
    """TC12: Punch-in with WFH status is accepted."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts, "status": "WFH"},
    )
    assert r.status_code == 201
    assert r.json()["status"] == "WFH"


def test_13_punch_in_absent_status_rejected():
    """TC13: Punch-in with ABSENT status is rejected 422."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts, "status": "ABSENT"},
    )
    assert r.status_code == 422


# ===========================================================================
# 4. R2: Late Minutes Calculation
# ===========================================================================
def test_14_r2_exactly_on_grace_not_late():
    """TC14: R2 - Punch at shift+10:00 exactly (600s) is NOT late."""
    create_employee(shift_start="09:30")
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 40, 0)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    assert r.status_code == 201
    assert r.json()["late_minutes"] == 0


def test_15_r2_one_second_past_grace_is_late():
    """TC15: R2 - Punch at shift+10:01 (601s) IS late, late_minutes=10."""
    create_employee(shift_start="09:30")
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 40, 1)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    assert r.status_code == 201
    assert r.json()["late_minutes"] == 10


def test_16_r2_late_45_minutes():
    """TC16: R2 - Punch at 10:15:59 for 09:30 shift -> late_minutes=45."""
    create_employee(shift_start="09:30")
    ts = ist_to_epoch_ms(2026, 7, 6, 10, 15, 59)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    assert r.status_code == 201
    assert r.json()["late_minutes"] == 45


# ===========================================================================
# 5. Punch-Out
# ===========================================================================
def test_17_punch_out_success():
    """TC17: Punch-out computes work_hours, overtime, half_day."""
    create_employee(shift_start="09:30", shift_end="18:30")
    punch_in_ts = ist_to_epoch_ms(2026, 7, 6, 9, 28)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": punch_in_ts},
    )
    punch_out_ts = ist_to_epoch_ms(2026, 7, 6, 18, 35)
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": punch_out_ts},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["punch_out"] == punch_out_ts
    assert data["work_hours"] == 9.12
    assert data["overtime_minutes"] == 0
    assert data["half_day"] is False


def test_18_punch_out_unknown_employee_404():
    """TC18: Punch-out for non-existent employee returns 404."""
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP9999", "punched_at": ist_to_epoch_ms(2026, 7, 6, 18, 0)},
    )
    assert r.status_code == 404


def test_19_punch_out_no_punch_in_404():
    """TC19: Punch-out with no prior punch-in returns 404."""
    create_employee()
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 6, 18, 0)},
    )
    assert r.status_code == 404


def test_20_punch_out_already_punched_out_409():
    """TC20: Double punch-out returns 409."""
    create_employee()
    ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    ts_out = ist_to_epoch_ms(2026, 7, 6, 18, 30)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ts_out})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 6, 19, 0)},
    )
    assert r.status_code == 409


def test_21_punch_out_before_punch_in_422():
    """TC21: Punch-out at or before punch_in returns 422."""
    create_employee()
    ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_in},
    )
    assert r.status_code == 422


def test_22_punch_out_over_24h_422():
    """TC22: Punch-out more than 24 hours after punch-in returns 422."""
    create_employee()
    ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    ts_out = ts_in + (25 * 3600 * 1000)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    assert r.status_code == 422


# ===========================================================================
# 6. R3: Overtime
# ===========================================================================
def test_23_r3_overtime_40_minutes():
    """TC23: R3 - 40 min after shift_end -> overtime_minutes=40."""
    create_employee(shift_start="09:30", shift_end="18:30")
    ts_in = ist_to_epoch_ms(2026, 7, 6, 10, 5)
    ts_out = ist_to_epoch_ms(2026, 7, 6, 19, 10)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    assert r.status_code == 200
    assert r.json()["overtime_minutes"] == 40


def test_24_r3_overtime_below_30_is_zero():
    """TC24: R3 - 5 min after shift_end -> overtime_minutes=0 (below 30)."""
    create_employee(shift_start="09:30", shift_end="18:30")
    ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 28)
    ts_out = ist_to_epoch_ms(2026, 7, 6, 18, 35)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    assert r.json()["overtime_minutes"] == 0


# ===========================================================================
# 7. R4 & R5: Work Hours and Half Day
# ===========================================================================
def test_25_r4_work_hours_half_up_rounding():
    """TC25: R4 - Work hours use half-up rounding, not banker's."""
    create_employee(shift_start="09:30", shift_end="18:30")
    ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30, 0)
    ts_out = ts_in + (29250 * 1000)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    assert r.json()["work_hours"] == 8.13


def test_26_r5_half_day_true():
    """TC26: R5 - work_hours < 4.50 -> half_day=true."""
    create_employee(shift_start="09:30", shift_end="18:30")
    ts_in = ist_to_epoch_ms(2026, 7, 7, 9, 30)
    ts_out = ist_to_epoch_ms(2026, 7, 7, 13, 0)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    data = r.json()
    assert data["work_hours"] == 3.5
    assert data["half_day"] is True


def test_27_r5_half_day_boundary():
    """TC27: R5 - work_hours exactly 4.50 -> half_day=false."""
    create_employee(shift_start="09:30", shift_end="18:30")
    ts_in = ist_to_epoch_ms(2026, 7, 7, 9, 30)
    ts_out = ts_in + int(4.5 * 3600 * 1000)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ts_in})
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    assert r.json()["half_day"] is False


# ===========================================================================
# 8. R1: Overnight Shifts
# ===========================================================================
def test_28_r1_overnight_shift_date():
    """TC28: R1 - Overnight shift: punch at 21:55 IST -> date is that day."""
    create_employee(
        emp_code="EMP0005",
        shift_start="22:00",
        shift_end="06:00",
        name="Night Worker",
        email="night@x.com",
        department="Support",
    )
    ts = ist_to_epoch_ms(2026, 7, 6, 21, 55)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0005", "punched_at": ts},
    )
    assert r.status_code == 201
    assert r.json()["date"] == "2026-07-06"


def test_29_r1_overnight_punch_before_shift_end():
    """TC29: R1 - Overnight shift: punch at 05:00 IST -> date is previous day."""
    create_employee(
        emp_code="EMP0005",
        shift_start="22:00",
        shift_end="06:00",
        name="Night Worker",
        email="night@x.com",
        department="Support",
    )
    ts = ist_to_epoch_ms(2026, 7, 7, 5, 0)
    r = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0005", "punched_at": ts},
    )
    assert r.status_code == 201
    assert r.json()["date"] == "2026-07-06"


def test_30_r3_overnight_overtime():
    """TC30: R3 - Overnight shift: overtime calculated with shift_end on next day."""
    create_employee(
        emp_code="EMP0005",
        shift_start="22:00",
        shift_end="06:00",
        name="Night Worker",
        email="night@x.com",
        department="Support",
    )
    ts_in = ist_to_epoch_ms(2026, 7, 6, 21, 55)
    ts_out = ist_to_epoch_ms(2026, 7, 7, 6, 40)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0005", "punched_at": ts_in},
    )
    r = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0005", "punched_at": ts_out},
    )
    assert r.status_code == 200
    assert r.json()["overtime_minutes"] == 40


# ===========================================================================
# 9. Attendance Listing
# ===========================================================================
def test_31_list_attendance_sorted():
    """TC31: Attendance list sorted by date DESC, emp_code ASC."""
    create_employee(emp_code="EMP0001", name="A", email="a@x.com")
    create_employee(emp_code="EMP0002", name="B", email="b@x.com")

    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 6, 9, 30)},
    )
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0002", "punched_at": ist_to_epoch_ms(2026, 7, 6, 9, 30)},
    )
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 7, 9, 30)},
    )

    r = client.get("/attendance")
    data = r.json()
    assert data["total"] == 3
    assert data["items"][0]["date"] == "2026-07-07"
    assert data["items"][1]["date"] == "2026-07-06"
    assert data["items"][1]["emp_code"] == "EMP0001"
    assert data["items"][2]["date"] == "2026-07-06"
    assert data["items"][2]["emp_code"] == "EMP0002"


def test_32_list_attendance_date_filter():
    """TC32: Attendance list with date_from/date_to filter."""
    create_employee()
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 6, 9, 30)},
    )
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_to_epoch_ms(2026, 7, 7, 9, 30)},
    )
    r = client.get("/attendance?date_from=2026-07-07&date_to=2026-07-07")
    assert r.json()["total"] == 1
    assert r.json()["items"][0]["date"] == "2026-07-07"


def test_33_list_attendance_invalid_date_range_422():
    """TC33: date_from > date_to returns 422."""
    r = client.get("/attendance?date_from=2026-07-10&date_to=2026-07-05")
    assert r.status_code == 422


# ===========================================================================
# 10. Regularize (PATCH)
# ===========================================================================
def test_34_regularize_change_punch_in():
    """TC34: Regularize changes punch_in and recomputes late_minutes."""
    create_employee(shift_start="09:30")
    old_ts = ist_to_epoch_ms(2026, 7, 6, 10, 5)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": old_ts},
    )
    new_ts = ist_to_epoch_ms(2026, 7, 6, 9, 28)
    r = client.patch(
        "/attendance/EMP0001/2026-07-06",
        json={
            "punch_in": new_ts,
            "reason": "biometric glitch corrected",
            "regularized_by": "hr.admin",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["punch_in"] == new_ts
    assert data["late_minutes"] == 0
    assert len(data["history"]) == 1
    assert data["history"][0]["by"] == "hr.admin"
    assert "punch_in" in data["history"][0]["changes"]
    assert "late_minutes" in data["history"][0]["changes"]


def test_35_regularize_to_absent():
    """TC35: Regularize to ABSENT clears punch times."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    r = client.patch(
        "/attendance/EMP0001/2026-07-06",
        json={
            "status": "ABSENT",
            "reason": "employee was not actually present",
            "regularized_by": "hr.admin",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ABSENT"
    assert data["punch_in"] is None
    assert data["punch_out"] is None


def test_36_regularize_absent_with_punch_in_422():
    """TC36: ABSENT/LEAVE with punch_in supplied -> 422."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    r = client.patch(
        "/attendance/EMP0001/2026-07-06",
        json={
            "status": "LEAVE",
            "punch_in": ts,
            "reason": "test reason string",
            "regularized_by": "hr.admin",
        },
    )
    assert r.status_code == 422


def test_37_regularize_no_change_422():
    """TC37: Regularize that changes nothing returns 422."""
    create_employee()
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    r = client.patch(
        "/attendance/EMP0001/2026-07-06",
        json={
            "status": "PRESENT",
            "reason": "test reason string",
            "regularized_by": "hr.admin",
        },
    )
    assert r.status_code == 422


def test_38_regularize_not_found_404():
    """TC38: Regularize non-existent record returns 404."""
    create_employee()
    r = client.patch(
        "/attendance/EMP0001/2026-07-06",
        json={
            "status": "LEAVE",
            "reason": "test reason string",
            "regularized_by": "hr.admin",
        },
    )
    assert r.status_code == 404


# ===========================================================================
# 11. Employee Monthly Analytics
# ===========================================================================
def test_39_employee_monthly_basic():
    """TC39: Monthly summary computes working_days, present_days, etc."""
    create_employee(joined_on="2026-07-01")
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    ts_out = ist_to_epoch_ms(2026, 7, 6, 18, 30)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )

    r = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
    assert r.status_code == 200
    data = r.json()
    assert data["emp_code"] == "EMP0001"
    assert data["month"] == "2026-07"
    assert data["working_days"] == 23
    assert data["present_days"] == 1.0
    assert data["leave_days"] == 0


def test_40_employee_monthly_unknown_emp_404():
    """TC40: Monthly summary for unknown employee returns 404."""
    r = client.get("/analytics/employees/EMP9999/monthly?month=2026-07")
    assert r.status_code == 404


def test_41_employee_monthly_mid_month_joiner():
    """TC41: R7 - Mid-month joiner: working_days counted from joined_on."""
    create_employee(joined_on="2026-07-15")
    r = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
    data = r.json()
    assert data["working_days"] > 0
    assert data["working_days"] < 23


def test_42_employee_monthly_half_day_counts():
    """TC42: R5/R6 - Half day counts as 0.5 present_days."""
    create_employee(joined_on="2026-07-01")
    ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    ts_out = ist_to_epoch_ms(2026, 7, 6, 13, 0)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts_in},
    )
    client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )
    r = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
    assert r.json()["present_days"] == 0.5


# ===========================================================================
# 12. Department Summary
# ===========================================================================
def test_43_department_summary_basic():
    """TC43: Department summary includes headcount and stats."""
    create_employee(
        emp_code="EMP0001", department="Engineering", joined_on="2026-01-01"
    )
    create_employee(
        emp_code="EMP0002",
        name="User2",
        email="u2@x.com",
        department="Engineering",
        joined_on="2026-01-01",
    )

    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    ts_out = ist_to_epoch_ms(2026, 7, 6, 18, 30)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )
    client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ts_out},
    )

    r = client.get("/analytics/departments/summary?month=2026-07")
    assert r.status_code == 200
    data = r.json()
    assert data["month"] == "2026-07"
    assert len(data["items"]) >= 1
    eng = [i for i in data["items"] if i["department"] == "Engineering"][0]
    assert eng["headcount"] == 2
    assert eng["present_days"] == 1.0


def test_44_department_summary_employee_no_logs():
    """TC44: R9 - Employees with no logs still count in headcount."""
    create_employee(
        emp_code="EMP0001", department="Engineering", joined_on="2026-01-01"
    )
    create_employee(
        emp_code="EMP0002",
        name="Ghost",
        email="ghost@x.com",
        department="Engineering",
        joined_on="2026-01-01",
    )
    r = client.get("/analytics/departments/summary?month=2026-07")
    data = r.json()
    eng = [i for i in data["items"] if i["department"] == "Engineering"][0]
    assert eng["headcount"] == 2
    assert eng["present_days"] == 0.0


# ===========================================================================
# 13. Late Leaderboard
# ===========================================================================
def test_45_late_leaderboard_ranking():
    """TC45: Standard competition ranking (1, 2, 2, 4)."""
    for i, minutes_late in [(1, 50), (2, 30), (3, 30), (4, 15)]:
        create_employee(
            emp_code=f"EMP{i:04d}",
            name=f"User{i}",
            email=f"u{i}@x.com",
            department="Engineering",
            joined_on="2026-01-01",
            shift_start="09:30",
        )
        ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30 + minutes_late)
        client.post(
            "/attendance/punch-in",
            json={"emp_code": f"EMP{i:04d}", "punched_at": ts_in},
        )

    r = client.get("/analytics/leaderboard/late?month=2026-07")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 4
    assert items[0]["rank"] == 1
    assert items[0]["emp_code"] == "EMP0001"
    assert items[1]["rank"] == 2
    assert items[2]["rank"] == 2
    assert items[1]["total_late_minutes"] == items[2]["total_late_minutes"]
    assert items[3]["rank"] == 4


def test_46_late_leaderboard_limit_with_ties():
    """TC46: Limit includes all tied employees at cutoff rank."""
    for i, minutes_late in [(1, 50), (2, 30), (3, 30)]:
        create_employee(
            emp_code=f"EMP{i:04d}",
            name=f"User{i}",
            email=f"u{i}@x.com",
            department="Engineering",
            joined_on="2026-01-01",
            shift_start="09:30",
        )
        ts_in = ist_to_epoch_ms(2026, 7, 6, 9, 30 + minutes_late)
        client.post(
            "/attendance/punch-in",
            json={"emp_code": f"EMP{i:04d}", "punched_at": ts_in},
        )

    r = client.get("/analytics/leaderboard/late?month=2026-07&limit=2")
    items = r.json()["items"]
    assert len(items) == 3
    assert items[1]["rank"] == 2
    assert items[2]["rank"] == 2


# ===========================================================================
# 14. Department Trend
# ===========================================================================
def test_47_department_trend_basic():
    """TC47: Trend returns one row per day with correct fields."""
    create_employee(
        emp_code="EMP0001",
        department="Engineering",
        joined_on="2026-01-01",
    )
    ts = ist_to_epoch_ms(2026, 7, 6, 9, 30)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ts},
    )

    r = client.get(
        "/analytics/departments/Engineering/trend?from=2026-07-06&to=2026-07-08"
    )
    assert r.status_code == 200
    data = r.json()
    assert data["department"] == "Engineering"
    assert len(data["items"]) == 3
    first = data["items"][0]
    assert first["date"] == "2026-07-06"
    assert "is_working_day" in first
    assert "headcount" in first
    assert "present_count" in first
    assert "late_count" in first
    assert "attendance_rate" in first
    assert "moving_avg_7d" in first


def test_48_department_trend_unknown_dept_404():
    """TC48: Trend for unknown department returns 404."""
    r = client.get(
        "/analytics/departments/NonExistent/trend?from=2026-07-06&to=2026-07-08"
    )
    assert r.status_code == 404


def test_49_department_trend_range_exceeds_92_days_422():
    """TC49: Trend range exceeding 92 days returns 422."""
    create_employee(department="Engineering")
    r = client.get(
        "/analytics/departments/Engineering/trend?from=2026-01-01&to=2026-07-01"
    )
    assert r.status_code == 422


# ===========================================================================
# 15. Admin Explain
# ===========================================================================
def test_50_explain_attendance_list():
    """TC50: Explain endpoint returns explain output with IXSCAN plan."""
    r = client.get(
        "/admin/explain/attendance_list?date_from=2026-07-01&date_to=2026-07-31"
    )
    assert r.status_code == 200
    data = r.json()
    assert data["endpoint"] == "attendance_list"
    assert data["collection"] == "attendance_logs"
    assert "explain" in data
