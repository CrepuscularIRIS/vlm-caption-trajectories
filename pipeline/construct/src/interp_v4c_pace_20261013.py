"""v4c: lower client concurrency for waves launched from 2026-10-13 16:10 UTC on.

User, 2026-10-13: "不要太高的并发, 不然磁盘会卡, 慢慢处理也可以". The workstation is shared and its disk
stalls under load, so new waves use 32 workers and 60 requests per minute instead of 48 and 90. Only the
values written into a wave's configuration at launch change; waves already launched keep their own.
"""
from pathlib import Path

import interp_v4r2_campaign_20261013 as v4r2

WORKERS, RPM = 32, 60


def apply() -> None:
    v4r2.base.WORKERS, v4r2.base.RPM = WORKERS, RPM
    v4r2.v4.WRAPPERS = (*v4r2.v4.WRAPPERS, Path(__file__).resolve())
