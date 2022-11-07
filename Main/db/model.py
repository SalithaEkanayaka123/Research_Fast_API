from sqlalchemy import Integer, String
from sqlalchemy.sql.schema import Column
from Main.db.database import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    hash_password = Column(String, nullable=False)

    # Add other necessary parameters for the user.
