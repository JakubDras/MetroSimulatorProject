from pydantic import BaseModel, Field
from typing import List, Tuple, Literal

# --- Definicje typów dla czytelności ---
Color = Tuple[int, int, int]
Position = Tuple[float, float]
Shape = Literal['circle', 'square', 'triangle', 'star', 'rombus']

# --- Modele Danych ---
class PassengerModel(BaseModel):
    origin_station_id: str
    target_shape: Shape
    travel_list: List[str] = Field(default_factory=list)

class StationModel(BaseModel):
    station_id: str
    pos: Position
    shape: Shape
    passengers: List[PassengerModel] = Field(default_factory=list)
    is_overcrowded: bool = False

class LineModel(BaseModel):
    color: Color
    station_ids: List[str] = Field(default_factory=list)
    is_loop: bool = False

class TrainModel(BaseModel):
    train_id: str
    line_color: Color
    current_station_id: str
    target_station_id: str
    pos: Position
    direction: int = 1
    passengers: List[PassengerModel] = Field(default_factory=list)