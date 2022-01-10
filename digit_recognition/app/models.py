from sqlalchemy import Column, Integer, String
from sqlalchemy.sql.sqltypes import Numeric

from .database import Base


class Images(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    image = Column(String)
    output = Column(Integer)
