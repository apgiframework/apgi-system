"""
Data Export Routes

API endpoints for exporting simulation data, summary statistics, and event analysis.
"""

import io
import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from api.models.schemas import ErrorResponse
from api.services.authorization import Permission, require_permission
from api.services.data_export import DataExportService

logger = logging.getLogger(__name__)


# Create router
router = APIRouter(
    prefix="/v1/sessions",
    tags=["Data Export"],
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# Global data export service instance
_data_export_service: Optional[DataExportService] = None


def get_data_export_service() -> DataExportService:
    """
    Dependency to get data export service instance.

    Returns:
        DataExportService instance
    """
    if _data_export_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data export service not initialized",
        )
    return _data_export_service


def init_export_routes(session_manager: Any) -> None:
    """
    Initialize export routes with session manager.

    Args:
        session_manager: SessionManager instance
    """
    global _data_export_service
    _data_export_service = DataExportService(session_manager)
    logger.info("Export routes initialized")


@router.get(
    "/{session_id}/export",
    summary="Export simulation data",
    description="Export complete simulation history in JSON or CSV format with optional filtering",
    dependencies=[Depends(require_permission(Permission.DATA_EXPORT))],
)
async def export_session_data(
    session_id: str,
    format: str = Query("json", description="Export format (json or csv)"),
    variables: Optional[str] = Query(
        None, description="Comma-separated list of variables to include"
    ),
    start_time: Optional[float] = Query(None, description="Start time for export (ms)"),
    end_time: Optional[float] = Query(None, description="End time for export (ms)"),
    service: DataExportService = Depends(get_data_export_service),
) -> StreamingResponse:
    """
    Export simulation data.

    Returns complete simulation history in the requested format (JSON or CSV).
    Supports filtering by time range and variable selection.

    Args:
        session_id: Unique session identifier
        format: Export format ('json' or 'csv')
        variables: Optional comma-separated list of variables to include
        start_time: Optional start time filter (milliseconds)
        end_time: Optional end time filter (milliseconds)
        service: Data export service dependency

    Returns:
        File download with appropriate Content-Type

    Raises:
        HTTPException: If session not found or export fails
    """
    try:
        # Parse variables list
        var_list = None
        if variables:
            var_list = [v.strip() for v in variables.split(",")]

        # Export data
        data_bytes, content_type = await service.export_session_data(
            session_id=session_id,
            format=format,
            variables=var_list,
            start_time=start_time,
            end_time=end_time,
        )

        # Determine filename
        extension = "json" if format.lower() == "json" else "csv"
        safe_session_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
        filename = f"session_{safe_session_id}_export.{extension}"

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(data_bytes),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except ValueError as e:
        logger.warning(f"Export failed for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    except Exception:
        logger.error(f"Failed to export data for session {session_id}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export data",
        )


@router.get(
    "/{session_id}/summary",
    summary="Get summary statistics",
    description="Retrieve computed summary statistics for simulation session",
    dependencies=[Depends(require_permission(Permission.DATA_READ))],
)
async def get_summary_statistics(
    session_id: str,
    service: DataExportService = Depends(get_data_export_service),
) -> Dict[str, Any]:
    """
    Get summary statistics.

    Returns computed metrics including:
    - Mean free energy
    - Ignition frequency and patterns
    - Metabolic efficiency
    - Allostatic load statistics

    Args:
        session_id: Unique session identifier
        service: Data export service dependency

    Returns:
        SummaryStatistics with computed metrics

    Raises:
        HTTPException: If session not found or computation fails
    """
    try:
        # Generate summary statistics
        stats = await service.generate_summary_stats(session_id)

        logger.info(f"Generated summary statistics for session {session_id}")
        return stats

    except ValueError as e:
        logger.warning(f"Summary generation failed for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    except Exception:
        logger.error(f"Failed to generate summary for session {session_id}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate summary statistics",
        )


@router.get(
    "/{session_id}/timeseries",
    summary="Get time series data",
    description="Retrieve timestamped sequences for specified variables with optional downsampling and pagination",
    dependencies=[Depends(require_permission(Permission.DATA_READ))],
)
async def get_time_series_data(
    session_id: str,
    variables: str = Query(..., description="Comma-separated list of variables"),
    start_time: Optional[float] = Query(None, description="Start time filter (ms)"),
    end_time: Optional[float] = Query(None, description="End time filter (ms)"),
    downsample: Optional[int] = Query(
        None, ge=1, description="Downsampling factor (every Nth point)"
    ),
    limit: Optional[int] = Query(
        None, ge=1, le=10000, description="Maximum number of data points to return"
    ),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    service: DataExportService = Depends(get_data_export_service),
) -> Dict[str, Any]:
    """
    Get time series data.

    Returns timestamped sequences for the specified variables with support for:
    - Time range filtering
    - Downsampling for large datasets
    - Cursor-based pagination

    Args:
        session_id: Unique session identifier
        variables: Comma-separated list of variable names
        start_time: Optional start time filter (milliseconds)
        end_time: Optional end time filter (milliseconds)
        downsample: Optional downsampling factor
        limit: Maximum number of data points to return
        cursor: Pagination cursor for fetching next page
        service: Data export service dependency

    Returns:
        Dict with time series data and pagination info

    Raises:
        HTTPException: If session not found or retrieval fails
    """
    try:
        # Parse variables list
        var_list = [v.strip() for v in variables.split(",")]

        # Get time series data
        result = await service.export_time_series(
            session_id=session_id,
            variables=var_list,
            start_time=start_time,
            end_time=end_time,
            downsample=downsample,
            limit=limit,
            cursor=cursor,
        )

        logger.info(
            f"Retrieved {result['num_points']} time series points " f"for session {session_id}"
        )
        return result

    except ValueError as e:
        logger.warning(f"Time series retrieval failed for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    except Exception:
        logger.error(f"Failed to get time series for session {session_id}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get time series data",
        )


@router.get(
    "/{session_id}/events",
    summary="Get event analysis",
    description="Retrieve aggregated statistics for ignition events including duration distribution and trigger patterns",
    dependencies=[Depends(require_permission(Permission.DATA_READ))],
)
async def get_event_analysis(
    session_id: str,
    event_type: str = Query("ignition", description="Type of event to analyze"),
    service: DataExportService = Depends(get_data_export_service),
) -> Dict[str, Any]:
    """
    Get event analysis.

    Returns aggregated statistics for the specified event type, including:
    - Total event count
    - Duration distribution (mean, min, max)
    - Trigger patterns and intervals

    Args:
        session_id: Unique session identifier
        event_type: Type of event to analyze (currently supports 'ignition')
        service: Data export service dependency

    Returns:
        Dict with event analysis statistics

    Raises:
        HTTPException: If session not found or analysis fails
    """
    try:
        # Get event analysis
        analysis = await service.get_event_analysis(session_id=session_id, event_type=event_type)

        logger.info(f"Generated {event_type} event analysis for session {session_id}")
        return analysis

    except ValueError as e:
        logger.warning(f"Event analysis failed for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    except Exception:
        logger.error(f"Failed to analyze events for session {session_id}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze events",
        )
