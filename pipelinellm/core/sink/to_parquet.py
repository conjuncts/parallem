from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl


def write_to_parquet(
    parquet_fpath: Path,
    data: Union[dict, list, pl.DataFrame],
    *,
    mode: Literal["append", "replace", "unique", "update"] = "append",
    on: Optional[list[str]] = None,
    receipt_col: Optional[list[str]] = None,
    schema=None,
) -> Optional[pl.DataFrame]:
    """
    Write data to a Parquet file.

    :param mode: How to write the data.
        Append: Adds new rows to the existing data.
        Replace: Replaces the existing data with the new data.
        Unique: Adds only new rows that don't exist in the existing data.
        Update: Updates existing rows with new data.
    :param on: Columns which collectively should serve as primary key.
        If mode is "unique", these are the columns ensured unique.
        If mode is "update", these are the columns used to identify rows to update.
    :param receipt_col: These column(s) will be taken from
        whatever is **newly added** to the dataframe.

    :return: The
    """
    if isinstance(data, pl.DataFrame):
        commit = data
    else:
        commit = pl.DataFrame(data, schema=schema)

    receipt_value = None
    parquet_fpath.parent.mkdir(parents=True, exist_ok=True)
    if parquet_fpath.exists():
        existing = pl.read_parquet(parquet_fpath)
        if mode in ["append", "unique"]:
            if mode == "unique":
                commit = commit.join(existing, on=on, how="anti")

            write_value = pl.concat([existing, commit], how="diagonal_relaxed")
        elif mode == "update":
            write_value = existing.update(commit, on=on)
        elif mode == "replace":
            write_value = commit
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        write_value = commit

    if receipt_col:
        receipt_value = commit.select(receipt_col)

    tmp_fpath = parquet_fpath.with_suffix(".tmp")
    write_value.write_parquet(tmp_fpath)
    tmp_fpath.replace(parquet_fpath)

    return receipt_value


class ParquetWriter:
    def __init__(self, parquet_fpath: Path, *, schema=None):
        """
        Manages a single Parquet file (akin to a table).

        :param schema: Optional schema for the Parquet file.
        """
        self.parquet_fpath = parquet_fpath
        self._log = []
        self.schema = schema

    def write(
        self,
        data: Union[dict, list, pl.DataFrame],
        mode: Literal["append", "replace", "unique", "update"],
        on: Optional[list[str]] = None,
        receipt_col: Union[str, list[str]] = None,
    ):
        return write_to_parquet(
            self.parquet_fpath, data, mode=mode, on=on, receipt_col=receipt_col
        )

    def log(self, item: dict):
        """Convenience method. See commit()."""
        self._log.append(item)

    def commit(
        self,
        mode: Literal["append", "replace", "unique", "update"],
        on: list[str] = None,
        receipt_col: Union[str, list[str]] = None,
    ):
        if self._log:
            ret = write_to_parquet(
                self.parquet_fpath,
                self._log,
                mode=mode,
                on=on,
                schema=self.schema,
                receipt_col=receipt_col,
            )
            self._log = []
            return ret
        return None

    def get(self, item: dict):
        """Retrieve items. Ignores any uncommitted items."""
        df = pl.read_parquet(self.parquet_fpath)
        query_df = pl.DataFrame([item])
        result_df = df.join(
            query_df, on=list(item.keys()), how="semi", nulls_equal=True
        )
        return result_df


class ParquetUniqueWriter(ParquetWriter):
    def __init__(self, parquet_fpath: Path, *, schema=None, unique_column_name: str):
        """
        Version of ParquetWriter that is more efficient for unique writes
        :param unique_column_name: Name of the unique column for unique writes
        """
        super().__init__(parquet_fpath, schema=schema)
        self._log_kv = {}
        self._unique_column_name = unique_column_name

    def log(self, item: dict):
        if self._unique_column_name not in item:
            raise ValueError(
                "Item must contain unique column", self._unique_column_name
            )
        key = item.pop(self._unique_column_name)
        self._log_kv[key] = item

    def log_kv(self, key: str, value: dict):
        """Convenience method. See commit()."""
        self._log_kv[key] = value

    def commit(
        self,
        mode: Literal["append", "replace", "unique", "update"],
        on: list[str] = None,
        receipt_col: Union[str, list[str]] = None,
    ):
        if self._log_kv:
            _log = [
                {self._unique_column_name: key, **value}
                for key, value in self._log_kv.items()
            ]
            for key, value in self._log_kv.items():
                ret = write_to_parquet(
                    self.parquet_fpath,
                    _log,
                    mode=mode,
                    on=on,
                    schema=self.schema,
                    receipt_col=receipt_col,
                )
            self._log = []
            return ret
        return None
