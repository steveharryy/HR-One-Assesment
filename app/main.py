"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import calendar
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), tz_aware=True)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))

MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.employees.create_index("emp_code", unique=True)
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)]
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)]
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", DESCENDING)]
    )
    db.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)]
    )
    db.attendance_logs.create_index(
        [("date", ASCENDING), ("late_minutes", ASCENDING)]
    )
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)]
    )
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)]
    )
    db.employees.create_index(
        [("joined_on", ASCENDING)]
    )
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan
)


def truncate_ms(dt: datetime) -> datetime:
    """R1: Truncate a datetime to whole seconds (drop sub-second precision)."""
    return dt.replace(microsecond=0)


def epoch_ms_to_dt(ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC-aware datetime, truncated to seconds."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return truncate_ms(dt)


def dt_to_epoch_ms(dt: datetime) -> int:
    """Convert a datetime to integer epoch milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def round_half_up(value: float, decimals: int) -> float:
    """Round using half-up rounding (R8)."""
    d = Decimal(str(value))
    quant = Decimal(10) ** -decimals
    return float(d.quantize(quant, rounding=ROUND_HALF_UP))


def compute_attendance_date(
    punch_in: datetime, shift_start: str, shift_end: str
) -> str:
    """R1: Attendance date is the IST calendar date of punch_in, with overnight
    shift adjustment.  An overnight shift has shift_end <= shift_start (e.g.
    22:00-06:00).  A punch-in whose IST time is *earlier than* shift_end belongs
    to the shift that started the previous calendar day."""
    punch_in_ist = punch_in.astimezone(IST)
    ist_date = punch_in_ist.date()

    sh, sm = map(int, shift_start.split(":"))
    eh, em = map(int, shift_end.split(":"))

    is_overnight = (eh * 60 + em) <= (sh * 60 + sm)

    if is_overnight:
        punch_secs = (
            punch_in_ist.hour * 3600
            + punch_in_ist.minute * 60
            + punch_in_ist.second
        )
        end_secs = eh * 3600 + em * 60
        if punch_secs < end_secs:
            ist_date -= timedelta(days=1)

    return ist_date.isoformat()


def compute_late_minutes(
    punch_in: datetime, shift_start: str, attendance_date: str
) -> int:
    """R2: Late if strictly more than 10:00 (600 s) after shift_start.
    late_minutes = floor(seconds since shift_start / 60).  Otherwise 0."""
    h, m = map(int, shift_start.split(":"))
    y, mo, d = map(int, attendance_date.split("-"))
    start_ist = datetime(y, mo, d, h, m, 0, tzinfo=IST)

    if punch_in.tzinfo is None:
        punch_in = punch_in.replace(tzinfo=timezone.utc)

    diff_seconds = (punch_in - start_ist).total_seconds()

    if diff_seconds <= 600:
        return 0

    return int(diff_seconds // 60)


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: (punch_out - punch_in) seconds / 3600, 2 decimals half-up."""
    seconds = (punch_out - punch_in).total_seconds()
    return round_half_up(seconds / 3600, 2)


def compute_overtime(
    punch_out: datetime,
    shift_end: str,
    attendance_date: str,
    shift_start: str,
) -> int:
    """R3: Overtime = whole minutes (floor) from shift_end to punch_out,
    counted only when >= 30; otherwise 0.  For overnight shifts the shift_end
    falls on the next calendar day."""
    eh, em = map(int, shift_end.split(":"))
    sh, sm = map(int, shift_start.split(":"))
    y, mo, d = map(int, attendance_date.split("-"))

    end_ist = datetime(y, mo, d, eh, em, 0, tzinfo=IST)

    if (eh * 60 + em) <= (sh * 60 + sm):
        end_ist += timedelta(days=1)

    if punch_out.tzinfo is None:
        punch_out = punch_out.replace(tzinfo=timezone.utc)

    diff_seconds = (punch_out - end_ist).total_seconds()
    if diff_seconds < 0:
        return 0

    minutes = int(diff_seconds // 60)
    return minutes if minutes >= 30 else 0


def compute_half_day(work_hours: float) -> bool:
    """R5: half_day is true when rounded work_hours < 4.50."""
    return work_hours < 4.50


def format_employee(doc: dict) -> dict:
    """Format a MongoDB employee document for the API response.
    Converts created_at to epoch ms, excludes _id."""
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_epoch_ms(doc["created_at"]),
    }


def format_attendance(doc: dict) -> dict:
    """Format a MongoDB attendance_log document for the API response.
    Converts punch times to epoch ms, defaults missing fields, excludes _id."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": (
            dt_to_epoch_ms(doc["punch_in"]) if doc.get("punch_in") else None
        ),
        "punch_out": (
            dt_to_epoch_ms(doc["punch_out"]) if doc.get("punch_out") else None
        ),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": format_history(doc.get("history", [])),
    }


def format_history(history: list) -> list:
    """Format history entries — BSON datetimes inside changes.punch_in /
    changes.punch_out must become epoch ms."""
    result = []
    for entry in history:
        formatted = {
            "at": dt_to_epoch_ms(entry["at"]),
            "by": entry["by"],
            "reason": entry["reason"],
            "changes": {},
        }
        for field, change in entry.get("changes", {}).items():
            if field in ("punch_in", "punch_out"):
                formatted["changes"][field] = {
                    "from": (
                        dt_to_epoch_ms(change["from"])
                        if change.get("from")
                        else None
                    ),
                    "to": (
                        dt_to_epoch_ms(change["to"])
                        if change.get("to")
                        else None
                    ),
                }
            else:
                formatted["changes"][field] = {
                    "from": change["from"],
                    "to": change["to"],
                }
        result.append(formatted)
    return result


class PresenceStatus(str, Enum):
    """Statuses valid for punch-in (presence statuses only)."""
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class StatusEnum(str, Enum):
    """All five attendance statuses."""
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class EmployeeIn(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("joined_on must be a valid date in YYYY-MM-DD format")
        return v

    @model_validator(mode="after")
    def shift_must_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None
    status: PresenceStatus = PresenceStatus.PRESENT

    @field_validator("punched_at")
    @classmethod
    def validate_epoch_ms(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < MIN_EPOCH_MS or v > MAX_EPOCH_MS):
            raise ValueError(
                f"punched_at must be between {MIN_EPOCH_MS} and {MAX_EPOCH_MS}"
            )
        return v


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("punched_at")
    @classmethod
    def validate_epoch_ms(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < MIN_EPOCH_MS or v > MAX_EPOCH_MS):
            raise ValueError(
                f"punched_at must be between {MIN_EPOCH_MS} and {MAX_EPOCH_MS}"
            )
        return v


class RegularizeIn(BaseModel):
    status: Optional[StatusEnum] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)

    @field_validator("punch_in", "punch_out")
    @classmethod
    def validate_epoch_ms(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < MIN_EPOCH_MS or v > MAX_EPOCH_MS):
            raise ValueError(
                f"time must be between {MIN_EPOCH_MS} and {MAX_EPOCH_MS}"
            )
        return v


def build_employee_monthly_pipeline(emp_code: str, month: str) -> list:
    year, mo = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, mo)[1]
    month_start_str = f"{month}-01"
    month_end_str = f"{month}-{days_in_month:02d}"

    return [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": month_start_str, "$lte": month_end_str},
            }
        },
        {
            "$project": {
                "date": 1,
                "status": 1,
                "half_day": 1,
                "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                "day_of_week": {
                    "$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                },
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]},
                                    {
                                        "$in": [
                                            "$status",
                                            ["PRESENT", "WFH", "ON_DUTY"],
                                        ]
                                    },
                                ]
                            },
                            {
                                "$cond": [
                                    {"$eq": ["$half_day", True]},
                                    0.5,
                                    1.0,
                                ]
                            },
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]


def build_department_summary_pipeline(
    month: str, department: Optional[str] = None
) -> list:
    year, mo = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, mo)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{days_in_month:02d}"

    match_filter: dict = {"joined_on": {"$lte": month_end}}
    if department:
        match_filter["department"] = department

    return [
        {"$match": match_filter},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {
                        "$match": {
                            "date": {"$gte": month_start, "$lte": month_end}
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": {"dept": "$department", "emp": "$emp_code"},
                "emp_present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {
                                        "$in": [
                                            {
                                                "$dayOfWeek": {
                                                    "$dateFromString": {
                                                        "dateString": "$logs.date"
                                                    }
                                                }
                                            },
                                            [2, 3, 4, 5, 6],
                                        ]
                                    },
                                    {
                                        "$in": [
                                            "$logs.status",
                                            ["PRESENT", "WFH", "ON_DUTY"],
                                        ]
                                    },
                                ]
                            },
                            {
                                "$cond": [
                                    {"$eq": ["$logs.half_day", True]},
                                    0.5,
                                    1.0,
                                ]
                            },
                            0.0,
                        ]
                    }
                },
                "emp_late_count": {
                    "$sum": {
                        "$cond": [{"$gt": ["$logs.late_minutes", 0]}, 1, 0]
                    }
                },
                "emp_total_late_minutes": {
                    "$sum": {"$ifNull": ["$logs.late_minutes", 0]}
                },
                "emp_leave_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]
                    }
                },
                "emp_on_duty_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]
                    }
                },
                "work_hours_list": {
                    "$push": {
                        "$cond": [
                            {
                                "$and": [
                                    {
                                        "$in": [
                                            "$logs.status",
                                            ["PRESENT", "WFH", "ON_DUTY"],
                                        ]
                                    },
                                    {"$ne": ["$logs.work_hours", None]},
                                ]
                            },
                            "$logs.work_hours",
                            "$$REMOVE",
                        ]
                    }
                },
            }
        },
        {
            "$group": {
                "_id": "$_id.dept",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": "$emp_present_days"},
                "late_count": {"$sum": "$emp_late_count"},
                "total_late_minutes": {"$sum": "$emp_total_late_minutes"},
                "leave_count": {"$sum": "$emp_leave_count"},
                "on_duty_count": {"$sum": "$emp_on_duty_count"},
                "all_work_hours": {"$push": "$work_hours_list"},
            }
        },
        {"$sort": {"_id": 1}},
    ]


def build_late_leaderboard_pipeline(
    month: str, department: Optional[str] = None
) -> list:
    year, mo = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, mo)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{days_in_month:02d}"

    pipeline = [
        {
            "$match": {
                "date": {"$gte": month_start, "$lte": month_end},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend(
        [
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "emp_code": "$_id",
                    "name": "$emp.name",
                    "department": "$emp.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ]
    )
    return pipeline


def build_department_trend_pipeline(
    department: str, from_date: str, to_date: str
) -> list:
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt_bound = to_dt + timedelta(days=1)

    dept_emps = list(
        db.employees.find({"department": department}, {"emp_code": 1})
    )
    emp_codes = [e["emp_code"] for e in dept_emps]

    return [
        {
            "$match": {
                "emp_code": {"$in": emp_codes},
                "date": {"$gte": from_date, "$lte": to_date},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {
                                "$cond": [
                                    {"$eq": ["$half_day", True]},
                                    0.5,
                                    1.0,
                                ]
                            },
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "date_dt": {"$dateFromString": {"dateString": "$_id"}},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$densify": {
                "field": "date_dt",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [from_dt, to_dt_bound],
                }
            }
        },
        {"$sort": {"date_dt": 1}},
        {
            "$set": {
                "date": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$date_dt"}
                },
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "day_of_week": {"$dayOfWeek": "$date_dt"},
            }
        },
        {
            "$set": {
                "is_working_day": {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]}
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"cur_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$cur_date"]},
                                ]
                            }
                        }
                    },
                    {"$count": "cnt"},
                ],
                "as": "hc_docs",
            }
        },
        {
            "$set": {
                "headcount": {
                    "$ifNull": [{"$arrayElemAt": ["$hc_docs.cnt", 0]}, 0]
                }
            }
        },
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                {"$eq": ["$is_working_day", True]},
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {
                            "$round": [
                                {"$divide": ["$present_count", "$headcount"]},
                                4,
                            ]
                        },
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date_dt": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, "current"]},
                    }
                },
            }
        },
        {
            "$set": {
                "moving_avg_7d": {
                    "$cond": [
                        {"$ne": ["$moving_avg_7d", None]},
                        {"$round": ["$moving_avg_7d", 4]},
                        None,
                    ]
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": 1,
            }
        },
    ]


@app.get("/health")
def health():
    """200 only when MongoDB answers a ping; otherwise 503."""
    try:
        client.admin.command("ping")
    except Exception:
        raise HTTPException(503, detail="MongoDB is not available")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, detail="emp_code already exists")
    doc.pop("_id", None)
    return format_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {}
    if department:
        q["department"] = department
    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    items = list(
        db.employees.find(q, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [format_employee(e) for e in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, detail="employee not found")

    if body.punched_at is not None:
        ts = epoch_ms_to_dt(body.punched_at)
    else:
        ts = truncate_ms(datetime.now(timezone.utc))

    att_date = compute_attendance_date(ts, emp["shift_start"], emp["shift_end"])

    late = compute_late_minutes(ts, emp["shift_start"], att_date)

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status.value,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, detail="already punched in for this date")

    doc.pop("_id", None)
    return format_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, detail="employee not found")

    if body.punched_at is not None:
        ts_out = epoch_ms_to_dt(body.punched_at)
    else:
        ts_out = truncate_ms(datetime.now(timezone.utc))

    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts_out}},
        sort=[("punch_in", DESCENDING)],
    )
    if rec is None:
        raise HTTPException(404, detail="no punch-in found")

    if rec.get("punch_out") is not None:
        raise HTTPException(409, detail="record is already punched out")

    diff_s = (ts_out - rec["punch_in"]).total_seconds()
    if diff_s <= 0 or diff_s > 24 * 3600:
        raise HTTPException(
            422,
            detail="punched_at must be strictly after punch_in and within 24 hours",
        )

    work_hours = compute_work_hours(rec["punch_in"], ts_out)
    overtime = compute_overtime(
        ts_out, emp["shift_end"], rec["date"], emp["shift_start"]
    )
    half_day = compute_half_day(work_hours)

    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts_out,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, detail="record is already punched out")

    return format_attendance(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusEnum] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, detail="date_from must not be after date_to")

    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status.value

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    docs = list(
        db.attendance_logs.find(q, {"_id": 0})
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )

    return {
        "items": [format_attendance(d) for d in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, detail="date must be in YYYY-MM-DD format")

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, detail="employee not found")

    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(404, detail="attendance record not found")

    old_status = rec["status"]
    old_punch_in = rec.get("punch_in")
    old_punch_out = rec.get("punch_out")
    old_work_hours = rec.get("work_hours")
    old_late_minutes = rec.get("late_minutes", 0)
    old_overtime_minutes = rec.get("overtime_minutes", 0)
    old_half_day = rec.get("half_day", False)

    target_status = body.status.value if body.status is not None else old_status

    if target_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                422,
                detail="supplying punch_in or punch_out with ABSENT or LEAVE is not allowed",
            )
        target_punch_in = None
        target_punch_out = None
        target_work_hours = None
        target_late = 0
        target_overtime = 0
        target_half_day = False
    else:
        if body.punch_in is not None:
            target_punch_in = epoch_ms_to_dt(body.punch_in)
        else:
            target_punch_in = old_punch_in

        if target_punch_in is None:
            raise HTTPException(422, detail="presence status requires punch_in")

        computed_date = compute_attendance_date(
            target_punch_in, emp["shift_start"], emp["shift_end"]
        )
        if computed_date != date:
            raise HTTPException(
                422, detail="punch_in must stay on the record's attendance date"
            )

        target_late = compute_late_minutes(
            target_punch_in, emp["shift_start"], date
        )

        if body.punch_out is not None:
            target_punch_out = epoch_ms_to_dt(body.punch_out)
        else:
            target_punch_out = old_punch_out

        if target_punch_out is not None:
            diff_s = (target_punch_out - target_punch_in).total_seconds()
            if diff_s <= 0 or diff_s > 24 * 3600:
                raise HTTPException(
                    422,
                    detail="punch_out must be strictly after punch_in and within 24 hours",
                )
            target_work_hours = compute_work_hours(
                target_punch_in, target_punch_out
            )
            target_overtime = compute_overtime(
                target_punch_out, emp["shift_end"], date, emp["shift_start"]
            )
            target_half_day = compute_half_day(target_work_hours)
        else:
            target_work_hours = None
            target_overtime = 0
            target_half_day = False

    changes = {}
    if old_status != target_status:
        changes["status"] = {"from": old_status, "to": target_status}
    if old_punch_in != target_punch_in:
        changes["punch_in"] = {"from": old_punch_in, "to": target_punch_in}
    if old_punch_out != target_punch_out:
        changes["punch_out"] = {"from": old_punch_out, "to": target_punch_out}
    if old_work_hours != target_work_hours:
        changes["work_hours"] = {"from": old_work_hours, "to": target_work_hours}
    if old_late_minutes != target_late:
        changes["late_minutes"] = {"from": old_late_minutes, "to": target_late}
    if old_overtime_minutes != target_overtime:
        changes["overtime_minutes"] = {
            "from": old_overtime_minutes,
            "to": target_overtime,
        }
    if old_half_day != target_half_day:
        changes["half_day"] = {"from": old_half_day, "to": target_half_day}

    if not changes:
        raise HTTPException(
            422, detail="a request that changes nothing is rejected"
        )

    history_entry = {
        "at": truncate_ms(datetime.now(timezone.utc)),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    old_history = rec.get("history", [])
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "history": old_history},
        {
            "$set": {
                "status": target_status,
                "punch_in": target_punch_in,
                "punch_out": target_punch_out,
                "work_hours": target_work_hours,
                "late_minutes": target_late,
                "overtime_minutes": target_overtime,
                "half_day": target_half_day,
            },
            "$push": {"history": history_entry},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(
            409, detail="concurrent edit conflict on attendance record"
        )

    return format_attendance(updated)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, detail="employee not found")

    year, mo = map(int, month.split("-"))
    days_in_month = calendar.monthrange(year, mo)[1]
    month_start_date = date(year, mo, 1)
    month_end_date = date(year, mo, days_in_month)

    joined_date = datetime.strptime(emp["joined_on"], "%Y-%m-%d").date()

    if joined_date > month_end_date:
        working_days = 0
    else:
        start_count = max(month_start_date, joined_date)
        working_days = 0
        curr = start_count
        while curr <= month_end_date:
            if curr.weekday() < 5:
                working_days += 1
            curr += timedelta(days=1)

    pipeline = build_employee_monthly_pipeline(emp_code, month)
    agg_res = list(db.attendance_logs.aggregate(pipeline))

    if agg_res:
        row = agg_res[0]
        present_days = float(row.get("present_days", 0.0))
        leave_days = int(row.get("leave_days", 0))
        late_count = int(row.get("late_count", 0))
        total_late = int(row.get("total_late_minutes", 0))
        total_ot = int(row.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late = 0
        total_ot = 0

    if working_days > 0:
        attendance_pct = round_half_up(
            (present_days / working_days) * 100, 2
        )
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_ot,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    pipeline = build_department_summary_pipeline(month, department)
    agg_res = list(db.employees.aggregate(pipeline))

    items = []
    for row in agg_res:
        all_wh = []
        for emp_list in row.get("all_work_hours", []):
            all_wh.extend(emp_list)

        if all_wh:
            avg_wh = round_half_up(sum(all_wh) / len(all_wh), 2)
        else:
            avg_wh = None

        items.append(
            {
                "department": row["_id"],
                "headcount": int(row.get("headcount", 0)),
                "present_days": float(row.get("present_days", 0.0)),
                "avg_work_hours": avg_wh,
                "late_count": int(row.get("late_count", 0)),
                "total_late_minutes": int(row.get("total_late_minutes", 0)),
                "leave_count": int(row.get("leave_count", 0)),
                "on_duty_count": int(row.get("on_duty_count", 0)),
            }
        )

    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    pipeline = build_late_leaderboard_pipeline(month, department)
    rows = list(db.attendance_logs.aggregate(pipeline))

    ranked_items = []
    current_rank = 1
    for idx, item in enumerate(rows):
        if (
            idx > 0
            and item["total_late_minutes"] < rows[idx - 1]["total_late_minutes"]
        ):
            current_rank = idx + 1

        if current_rank > limit:
            break

        ranked_items.append(
            {
                "rank": current_rank,
                "emp_code": item["emp_code"],
                "name": item["name"],
                "department": item["department"],
                "total_late_minutes": int(item["total_late_minutes"]),
                "late_count": int(item["late_count"]),
            }
        )

    return {"month": month, "items": ranked_items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: str = Query(..., alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    try:
        d_from = datetime.strptime(from_date, "%Y-%m-%d").date()
        d_to = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(422, detail="invalid date format")

    if d_to < d_from:
        raise HTTPException(
            422, detail="'to' date must not be before 'from' date"
        )

    if (d_to - d_from).days > 92:
        raise HTTPException(422, detail="date range must not exceed 92 days")

    dept_count = db.employees.count_documents({"department": department})
    if dept_count == 0:
        raise HTTPException(404, detail="Department not found")

    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    rows = list(db.attendance_logs.aggregate(pipeline))

    items = []
    for r in rows:
        att_rate = r.get("attendance_rate")
        if att_rate is not None:
            att_rate = round_half_up(float(att_rate), 4)

        mov_avg = r.get("moving_avg_7d")
        if mov_avg is not None:
            mov_avg = round_half_up(float(mov_avg), 4)

        items.append(
            {
                "date": r["date"],
                "is_working_day": r["is_working_day"],
                "headcount": int(r["headcount"]),
                "present_count": float(r["present_count"]),
                "late_count": int(r["late_count"]),
                "attendance_rate": att_rate,
                "moving_avg_7d": mov_avg,
            }
        )

    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusEnum] = None,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, detail="date_from must not be after date_to")

        q: dict = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status.value

        skip = (page - 1) * page_size
        explain_res = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": q,
                "sort": {"date": -1, "emp_code": 1},
                "skip": skip,
                "limit": page_size,
            },
            verbosity="executionStats",
        )

    elif endpoint == "employee_monthly":
        collection = "attendance_logs"
        if not emp_code or not month:
            raise HTTPException(
                422, detail="emp_code and month are required for employee_monthly"
            )
        pipeline = build_employee_monthly_pipeline(emp_code, month)
        explain_res = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )

    elif endpoint == "department_summary":
        collection = "employees"
        if not month:
            raise HTTPException(
                422, detail="month is required for department_summary"
            )
        pipeline = build_department_summary_pipeline(month, department)
        explain_res = db.command(
            "explain",
            {
                "aggregate": "employees",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )

    elif endpoint == "late_leaderboard":
        collection = "attendance_logs"
        if not month:
            raise HTTPException(
                422, detail="month is required for late_leaderboard"
            )
        pipeline = build_late_leaderboard_pipeline(month, department)
        explain_res = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )

    elif endpoint == "department_trend":
        collection = "attendance_logs"
        if not department or not from_date or not to_date:
            raise HTTPException(
                422,
                detail="department, from, and to are required for department_trend",
            )
        try:
            d_from = datetime.strptime(from_date, "%Y-%m-%d").date()
            d_to = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(422, detail="invalid date format")

        if d_to < d_from:
            raise HTTPException(
                422, detail="'to' date must not be before 'from' date"
            )
        if (d_to - d_from).days > 92:
            raise HTTPException(
                422, detail="date range must not exceed 92 days"
            )

        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        explain_res = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )

    else:
        raise HTTPException(
            422, detail=f"unknown endpoint for explain: {endpoint}"
        )

    return {
        "endpoint": endpoint,
        "collection": collection,
        "explain": jsonable_encoder(explain_res),
    }
