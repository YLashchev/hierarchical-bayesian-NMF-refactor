"""
Data schema for flexible panel data loading.

This module provides a schema-based approach to loading any panel data
without hardcoded column names.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class OutcomeSpec:
    """
    Specification for an outcome variable and its optional denominator.
    
    Attributes
    ----------
    outcome_col : str
        Column name for the outcome variable (e.g., "births_usborn")
    label : str
        Label to use for this outcome group (e.g., "usborn")
    denominator_col : Optional[str]
        Column name for the denominator (e.g., "pop_usborn")
    """
    outcome_col: str
    label: str
    denominator_col: Optional[str] = None


@dataclass
class DataSchema:
    """
    Schema defining how to interpret panel data columns.
    
    This allows the package to work with any panel data by specifying
    column mappings in configuration rather than code.
    
    Attributes
    ----------
    unit_col : str
        Column name for panel unit (e.g., "state", "firm", "county")
    time_col : str
        Column name for time period
    treatment_col : str
        Column name for treatment indicator (0/1)
    outcomes : List[OutcomeSpec]
        List of outcome specifications
    date_format : str
        Date parsing format ("auto" for automatic detection)
    additional_cols : List[str]
        Additional columns to preserve (e.g., "banned_state", "year", "month")
    """
    unit_col: str
    time_col: str
    treatment_col: str
    outcomes: List[OutcomeSpec]
    date_format: str = "auto"
    additional_cols: List[str] = field(default_factory=list)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DataSchema':
        """
        Create schema from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration with 'data' section containing 'schema' subsection
            
        Returns
        -------
        DataSchema
            Constructed schema object
        """
        schema_cfg = config['data']['schema']
        
        # Parse outcomes list
        outcomes = []
        for o in schema_cfg['outcomes']:
            outcomes.append(OutcomeSpec(
                outcome_col=o['outcome_col'],
                label=o['label'],
                denominator_col=o.get('denominator_col')
            ))
        
        return cls(
            unit_col=schema_cfg['unit_col'],
            time_col=schema_cfg['time_col'],
            treatment_col=schema_cfg['treatment_col'],
            outcomes=outcomes,
            date_format=config['data'].get('date_format', 'auto'),
            additional_cols=schema_cfg.get('additional_cols', [])
        )
    
    def get_outcome_cols(self) -> List[str]:
        """Get list of outcome column names."""
        return [o.outcome_col for o in self.outcomes]
    
    def get_denominator_cols(self) -> List[str]:
        """Get list of denominator column names (excluding None)."""
        return [o.denominator_col for o in self.outcomes if o.denominator_col]
    
    def get_labels(self) -> List[str]:
        """Get list of outcome labels (group names)."""
        return [o.label for o in self.outcomes]
    
    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate that dataframe matches schema.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
            
        Raises
        ------
        ValueError
            If required columns are missing
        """
        required = [self.unit_col, self.time_col, self.treatment_col]
        required.extend(self.get_outcome_cols())
        required.extend(self.get_denominator_cols())
        required.extend(self.additional_cols)
        
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Check your schema configuration."
            )
    
    def get_required_columns(self) -> List[str]:
        """Get all required column names."""
        cols = [self.unit_col, self.time_col, self.treatment_col]
        cols.extend(self.get_outcome_cols())
        cols.extend(self.get_denominator_cols())
        cols.extend(self.additional_cols)
        return cols


def create_simple_schema(
    unit_col: str = "state",
    time_col: str = "time",
    treatment_col: str = "treated",
    outcome_col: str = "outcome",
    denominator_col: Optional[str] = None,
    label: str = "total"
) -> DataSchema:
    """
    Create a simple schema for single-outcome panel data.
    
    Parameters
    ----------
    unit_col : str
        Panel unit column name
    time_col : str
        Time column name
    treatment_col : str
        Treatment indicator column name
    outcome_col : str
        Outcome column name
    denominator_col : str, optional
        Denominator column name
    label : str
        Label for the outcome group
        
    Returns
    -------
    DataSchema
        Simple schema with one outcome
    """
    return DataSchema(
        unit_col=unit_col,
        time_col=time_col,
        treatment_col=treatment_col,
        outcomes=[OutcomeSpec(
            outcome_col=outcome_col,
            label=label,
            denominator_col=denominator_col
        )]
    )
