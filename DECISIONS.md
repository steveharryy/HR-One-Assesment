# DECISIONS.md

1. **Indexes.**
- `employees.emp_code` (unique): Guarantees unique employee business codes.
- `attendance_logs(emp_code, date)` (unique compound): Enforces at most one record per employee per attendance day; blocks concurrent duplicate punch-ins.
- `attendance_logs(date DESC, emp_code ASC)`: Powers `GET /attendance` sort and range queries.
- `attendance_logs(emp_code, punch_in DESC)`: Enables index-backed lookup of the latest punch-in during punch-out.
- `attendance_logs(date, late_minutes)`: Powers `late_leaderboard` aggregation matching date ranges with `late_minutes > 0`.
- `employees(department, joined_on)` & `employees(joined_on)`: Speeds up R9 headcount checks for summary and trend.
*Rejected:* An index on `(punch_in, punch_out)` was rejected because punches are always queried per employee, making `(emp_code, punch_in)` far more selective.

2. **Punch-in race.**
Both concurrent requests calculate the IST attendance date and issue `insert_one()`. MongoDB's unique index on `(emp_code, date)` atomically accepts the first write and throws `DuplicateKeyError` on the second. The winner gets `201 Created`; the loser catches the error and returns `409 Conflict`.

3. **Ties.**
Standard competition ranking (1, 2, 2, 4) is applied. Per the contract, `limit` filters by `rank <= limit`. If employees tie at the cutoff, all tied rows are returned (possibly exceeding `limit` rows). Rows with identical late minutes are deterministically ordered by `emp_code` ascending.

4. **Headcount.**
The aggregation pipeline queries `employees` matching `joined_on <= month_end`, then performs `$lookup` into `attendance_logs` with `$unwind: { preserveNullAndEmptyArrays: true }`. Grouping by department counts employee records (`$sum: 1`), ensuring employees with zero logs are counted.

5. **One thing you would change for 100x data.**
I would maintain pre-aggregated daily/monthly summary collections via change streams or event workers rather than computing multi-collection `$lookup` and `$densify` pipelines on raw logs during read requests.
