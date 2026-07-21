import pandas as pd

from SRC.collectors.health import collect_health
from SRC.collectors.education import collect_education
from SRC.collectors.agriculture import collect_agriculture
from SRC.collectors.security import collect_security
from SRC.collectors.governance import collect_governance


def collect_all():

    records = []

    records.extend(collect_health())
    records.extend(collect_education())
    records.extend(collect_agriculture())
    records.extend(collect_security())
    records.extend(collect_governance())

    return pd.DataFrame(records)