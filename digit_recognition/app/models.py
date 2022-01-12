from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.sql.sqltypes import Numeric

from .database import Base


class Images(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String)
    prediction = Column(Float[0])
    probability = Column(Numeric[10, 2])
