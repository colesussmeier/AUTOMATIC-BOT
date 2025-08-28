import os
import aiohttp
import asyncio
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List


class PredictionMarketSearchClient:
    def __init__(self):
        load_dotenv()
        self.API_KEY = os.getenv('ADJ_API_KEY')
        self.BASE_URL = "https://api.data.adj.news/api"
        self.SEARCH_URL = "/search/query"
        self._session = None
        self.adj_results = []
        self.manifold_results = []
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("Session not initialized. Use 'async with' context manager or call __aenter__() first.")
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
    
    def request_adjacent_headers(self) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        return headers
    
    async def adjacent_semantic_search(
            self, 
            query: str, 
            limit: Optional[int] = 25,
            include_context: Optional[bool] = False,
            filter_future_only: Optional[bool] = True,
            minimum_volume: Optional[float] = None
            ) -> List[Dict[str, Any]]:
        
        params = {}

        params["q"] = query
        params["limit"] = limit
        params["include_context"] = str(include_context).lower()
        
        try:
            session = await self._get_session()
            async with session.get(
                url = f"{self.BASE_URL}{self.SEARCH_URL}",
                headers = self.request_adjacent_headers(),
                params = params
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if filter_future_only:
                    return self._filter_markets(response_data, minimum_volume)
                else:
                    return response_data.get('data', []) if isinstance(response_data, dict) else []
                
        except Exception as e:
            print(f"Error fetching data from Adj: {e}")
            return []
    
    # Filter for unresolved (live) markets and markets with sufficient volume
    def _filter_markets(self, data: Dict[str, Any], minimum_volume: Optional[float] = None) -> List[Dict[str, Any]]:
        try:
            if 'data' in data and isinstance(data['data'], list):
                today = datetime.now(timezone.utc)
                filtered_data = []
                fields = ["market_id", "platform", "market_type", "question", "description", "rules", "probability", "volume"]
                
                for market in data['data']:
                    try:
                        end_date_str = market.get('end_date')
                        if not end_date_str:
                            continue
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        if end_date <= today:
                            continue
                        
                        # Filter by minimum volume if specified (but not for Metaculus)
                        if minimum_volume is not None:
                            platform = market.get('platform', '').lower()
                            # Metaculus has no volume so don't filter it
                            if platform != 'metaculus':
                                market_volume = market.get('volume')
                                if market_volume is None or float(market_volume) < minimum_volume:
                                    continue
                        
                        filtered_market = {key: market.get(key) for key in fields if key in market}
                        filtered_data.append(filtered_market)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Skipping market due to date parsing error: {e}")
                        continue
                
                # Get at most 10 markets
                return filtered_data[:10]
                
        except Exception as e:
            print(f"Error filtering markets: {e}")
            return []
    
    # Similar to semantic search for Adjacent
    async def search_manifold(
        self,
        term: str,
        sort: Optional[str] = "most-popular",
        filter: Optional[str] = "open",
        limit: Optional[int] = 5,
        liquidity: Optional[int] = 1000,
        minimum_volume: Optional[float] = None
    ) -> List[str]:
        try:
            url = "https://api.manifold.markets/v0/search-markets"

            params = {
                "term": term,
                "sort": sort,
                "limit": limit,
                "filter": filter,
                "liquidity": liquidity
            }
            
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    mkts = []

                    for item in data:
                        if (item["outcomeType"] == 'BINARY') or (item["outcomeType"] == 'MULTIPLE_CHOICE'):
                            if minimum_volume is None or item["volume"] > minimum_volume:
                                mkt = item["id"]
                                mkts.append(mkt)
                    
                    return mkts
            
        except Exception as e:
            print(f"Error searching Manifold Markets: {e}")
            return []

    # Search individual markets to get details    
    async def search_manifold_market(self, id: str) -> Dict[str, Any]:
        try:
            url = f"https://api.manifold.markets/v0/market/{id}"
            
            session = await self._get_session()
            async with session.get(url) as response:
                    response.raise_for_status()
                    mkt = await response.json()

                    if mkt["outcomeType"] == 'BINARY':
                        return {
                            "id": mkt["id"],
                            "question": mkt["question"], 
                            "probability": round(mkt["probability"] * 100),
                            "volume": mkt["volume"],
                            "description": mkt["textDescription"]
                        }
                    elif mkt["outcomeType"] == 'MULTIPLE_CHOICE':
                        return {
                            "id": mkt["id"],
                            "question": mkt["question"],
                            "probabilities_of_different_outcomes": {answer["text"]: round(answer["probability"] * 100) for answer in mkt["answers"]},
                            "volume": mkt["volume"],
                            "description": mkt["textDescription"]
                        }

        except Exception as e:
            print(f"Error searching Manifold Market Probabilities: {e}")
            return {}
        
    
    async def search_prediction_markets(
            self, 
            query: str, 
            minimum_volume: Optional[float] = None, 
            is_first_search = True
        ) -> List[List[Dict[str, Any]]]:
        # Run both searches concurrently
        adj_task = self.adjacent_semantic_search(query, minimum_volume=minimum_volume)
        manifold_ids_task = self.search_manifold(query, minimum_volume=minimum_volume)
        adj, manifold_ids = await asyncio.gather(adj_task, manifold_ids_task)
        
        manifold = []

        if is_first_search:
            for id in manifold_ids:
                mkt = await self.search_manifold_market(id)
                manifold.append(mkt)

            self.adj_results.extend(adj)
            self.manifold_results.extend(manifold)

        # If it is not the first search, only add new results (this avoids overlaps)
        else:
            existing_adj_ids = [mkt["market_id"] for mkt in self.adj_results]
            for mkt in adj:
                if mkt["market_id"] not in existing_adj_ids:
                    self.adj_results.append(mkt)

            existing_manifold_ids = [mkt["id"] for mkt in self.manifold_results]
            for id in manifold_ids:
                if id not in existing_manifold_ids:
                    mkt = await self.search_manifold_market(id)
                    self.manifold_results.append(mkt)

        await asyncio.sleep(2)
        return [self.adj_results, self.manifold_results]