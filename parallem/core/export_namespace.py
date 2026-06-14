from typing import TYPE_CHECKING, Literal, Optional


if TYPE_CHECKING:
    from parallem.core.agent.orchestrator import AgentOrchestrator
    import polars as pl

class ExportNamespace:
    """Namespace for export-related methods."""

    def __init__(self, orch: "AgentOrchestrator"):
        self._orch = orch

    def to_directory(
        self,
        directory: Optional[str],
        *,
        filetype: Literal["csv", "tsv", "parquet"] = "parquet",
    ):
        """
        Export all tables from the backend to files.

        :param directory: Directory to export tables to. If None, uses the default datastore directory.
        :param filetype: Export file type - "polars" or "parquet" for parquet files, "csv" for CSV, "tsv" for TSV.
        """

        directory.mkdir(parents=True, exist_ok=True)

        for table_name, df in self.to_polars().items():
            fpath = directory / f"{table_name}.{filetype}"

            if filetype == "parquet":
                df.write_parquet(fpath)
            elif filetype == "csv":
                df.write_csv(fpath)
            elif filetype == "tsv":
                df.write_csv(fpath, separator="\t")

    def to_polars(
        self,
    ) -> dict[str, "pl.DataFrame"]:
        """
        Export all tables from the backend as Polars DataFrames.

        :returns: A dictionary mapping table names to Polars DataFrames.
        """
        return self._orch._backend._get_datastore().export_polars()
