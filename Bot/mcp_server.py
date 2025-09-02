import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient
from typing import List, Optional, Dict, Tuple
from bson import ObjectId
from datetime import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import random
import numpy as np
import math
from scipy.spatial import Delaunay
from mcp.server.fastmcp.utilities.types import Image
from matplotlib import cm
from mcp.types import ImageContent
import matplotlib.colors as mcolors
import base64
# from mcp.types import ImageContent
from io import BytesIO

load_dotenv()

broadwaypass_mcp_server = FastMCP("BroadwayPass")
broadwaypass_mcp_server.settings.port = 8650

masterapi_mongo_uri = os.getenv("MASTERAPI_MONGO_URI")
masterapi_db_name = os.getenv("MASTERAPI_DB")
configuration_collection = os.getenv("CONFIG")
show_meta_info_collection = os.getenv("SHOW_META_INFO")
show_availability_collection = os.getenv("SHOW_AVAILABILITY")
seat_section_collection = os.getenv("SECTION_CHART")

masterapi_db_client = MongoClient(masterapi_mongo_uri)
masterapi_db = masterapi_db_client[masterapi_db_name]


@broadwaypass_mcp_server.tool()
async def fetch_all_available_shows(
        active_only: bool = True,
        additional_fields: Optional[List[str]] = None
) -> List[Dict]:
    """
        Fetches show information from the configuration collection with flexible querying options.

        This method retrieves show names, IDs, and optionally specified additional fields from the
        configuration document. It supports filtering by active status and can include any nested fields
        from the show's data structure using dot notation.

        Parameters
        ----------
        active_only : bool, optional
            If True (default), returns only active shows (where `isActive` is True or string "true"/"1").
            If False, returns all shows regardless of active status.
        additional_fields : List[str], optional
            List of additional fields to include in results. Use dot notation for nested fields
            (e.g., `showInfo.runningTime`, `platform`, `promocodes.CHCORP22.isActive`).
            Common optional fields in the database include:
            - `showInfo.runningTime`: Runtime duration of the show.
            - `showInfo.genre`: Genre classification.
            - `showInfo.venue`: Venue name.
            - `showInfo.address`: Venue address.
            - `platform`: Ticket platform (e.g., "telecharge", "broadwaydirect").
            - `thumbnail`: URL for show thumbnail image.
            - `coverImage`: URL for cover image.
            - `updatesIn`: Update frequency indicator.
            - `promocodes.<code>.isActive`: Status of specific promo codes.
            - `seatMapUrl`: URL for seat map (if available).
            - `urlPattern`: URL pattern for ticket links.
            Defaults to None (only returns show_id and show_name).

        Returns
        -------
        List[Dict]
            List of dictionaries containing:
            - `show_id` (str): Unique identifier for the show.
            - `show_name` (str): Name of the show.
            - Additional fields as specified in `additional_fields` (e.g., `showInfo.venue`, `platform`).
            Returns empty list if no matches found or document doesn't exist.

        Notes
        -----
        - The configuration document is identified by a fixed ObjectId ("66db3ed1cea1c555770a9f05").
        - Handles both boolean and string representations of `isActive` (e.g., True, "true", "1").
        - Nested fields are accessed via dot notation (e.g., "showInfo.address").
        - Missing optional fields return `None` in the results.

        Raises
        ------
        Exception
            Logs database errors internally but returns empty list on failure to maintain client stability.
        """
    try:
        collection = masterapi_db[configuration_collection]
        res = []

        # Fetch the configuration document using its ObjectId
        doc = collection.find_one({"_id": ObjectId("66db3ed1cea1c555770a9f05")})

        if not doc or "shows" not in doc:
            return []

        shows_data = doc["shows"]

        for show_id, show_info in shows_data.items():
            # Check active status filter
            if active_only:
                is_active = show_info.get("isActive")
                # Handle both boolean and string representations
                if isinstance(is_active, str):
                    if is_active.lower() not in ["true", "1"]:
                        continue
                elif not isinstance(is_active, bool) or not is_active:
                    continue

            # Base result with required fields
            result = {
                "show_id": show_id,
                "show_name": show_info.get("showName", "")
            }

            # Add optional fields if specified
            if additional_fields:
                for field in additional_fields:
                    # Handle nested fields using dot notation
                    keys = field.split('.')
                    value = show_info
                    for key in keys:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            value = None
                            break
                    result[field] = value

            res.append(result)

        return res

    except Exception as e:
        # Log error details for debugging
        print(f"Database error in fetch_all_available_shows: {str(e)}")
        return []


@broadwaypass_mcp_server.tool()
async def fetch_show_calendar(
        show_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_sold_out: bool = True
) -> List[Dict]:
    """
    Fetches show calendar information (dates and times) from the showMetaInfo collection.

    This method retrieves performance schedules for specified shows, with options to filter
    by date range and sold-out status. Returns a condensed format with show information
    and a dictionary of performances grouped by date.

    Parameters:
        show_ids (List[str]): Optional list of specific show IDs to fetch calendar for.
        start_date (str): Optional start date for filtering in DD-MM-YYYY format.
        end_date (str): Optional end date for filtering in DD-MM-YYYY format.
        include_sold_out (bool): If False, filters out sold-out performances. Defaults to True.

    Returns:
        List[Dict]: A list of shows with their performance schedules in a condensed format:
        {
            "show_name": str,
            "theater_address": str,
            "performances": {
                "date": {
                    "time": "perf_key",
                    ...
                },
                ...
            }
        }
    """
    try:
        collection = masterapi_db[show_meta_info_collection]
        query = {}

        # Build query based on provided parameters
        if show_ids:
            query["showId"] = {"$in": show_ids}

        # Fetch documents matching the query
        docs = collection.find(query)
        results = []

        for doc in docs:
            if "availability" not in doc:
                continue

            # Create show entry with basic info
            show_entry = {
                "show_name": doc.get("showName", ""),
                "theater_address": doc.get("showInfo", {}).get("address", ""),
                "performances": {}
            }

            # Process each date in the availability calendar
            for date_str, date_info in doc["availability"].items():
                # Skip metadata fields
                if date_str in ["updatesIn", "lastUpdated"]:
                    continue

                # Apply date range filter if specified
                if start_date and end_date:
                    try:
                        current_date = datetime.strptime(date_str, "%d-%m-%Y")
                        start_dt = datetime.strptime(start_date, "%d-%m-%Y")
                        end_dt = datetime.strptime(end_date, "%d-%m-%Y")

                        if not (start_dt <= current_date <= end_dt):
                            continue
                    except ValueError:
                        # Skip invalid date formats
                        continue

                # Initialize date entry
                date_performances = {}

                # Process each time slot for this date
                for time_slot, performance_info in date_info.items():
                    if time_slot in ["updatesIn", "lastUpdated"]:
                        continue

                    # Skip sold-out performances if requested
                    if not include_sold_out and performance_info.get("soldOut", 0) == 1:
                        continue

                    # Add time slot and performance key
                    date_performances[time_slot] = performance_info.get("perfKey")

                # Only add date if there are performances
                if date_performances:
                    show_entry["performances"][date_str] = date_performances

            # Only add show if there are performances
            if show_entry["performances"]:
                results.append(show_entry)

        return results

    except Exception as e:
        print(f"Database error in fetch_show_calendar: {str(e)}")
        return []





@broadwaypass_mcp_server.tool()
async def generate_section_data(perfKey: str) -> Dict:
    """
    Generates a JSON object containing section information for a specific theater performance.

    Parameters
    ----------
    perfKey : str
        Unique identifier for a performance (from fetch_show_calendar).

    Returns
    -------
    Dict
        JSON object containing:
        - sections: List of section objects with name, price, and available seats
        - performance_key: The input performance identifier
    """
    try:
        # Get showAvailability document
        availability_collection = masterapi_db[show_availability_collection]
        availability_doc = availability_collection.find_one({"perfKey": perfKey})

        if not availability_doc:
            return {"error": "No availability data found for the given performance key"}

        show_id = availability_doc["showId"]
        seats = availability_doc.get("seats", [])

        # Get sectionChart documents for this show
        section_collection = masterapi_db[seat_section_collection]
        sections = list(section_collection.find({"showId": show_id}))

        if not sections:
            return {"error": "No sections found for this show"}

        # Process each section
        section_data = []
        for section in sections:
            section_name = section.get("sectionName", "Unknown Section")
            price = section.get("price", 0)
            seat_ids = section.get("seatIds", [])

            # Count available seats (with price and not booked)
            available_seats = 0
            for seat in seats:
                if seat.get("id") in seat_ids:
                    has_price = "price" in seat and seat["price"] is not None
                    # is_available = not seat.get("isBooked", False)
                    if has_price:
                        available_seats += 1

            section_data.append({
                "sectionName": section_name,
                "price": price,
                "availableSeats": available_seats
            })

        return {
            "performanceKey": perfKey,
            "sections": section_data
        }

    except Exception as e:
        print(f"Error generating section data: {e}")
        return {"error": f"Error generating section data: {str(e)}"}


# for testing
async def main():
    data = await generate_section_data(perfKey="966764")
    print(data)

if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main())
    broadwaypass_mcp_server.run(transport="sse")