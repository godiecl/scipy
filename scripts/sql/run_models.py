#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import os

from logger import configure_logging
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typeguard import typechecked

# declare a base model
Base = declarative_base()


class Persona(Base):
    __tablename__ = "personas"
    id = Column(Integer, primary_key=True)
    nombre = Column(String, nullable=False)
    apellidos = Column(String, nullable=False)

    def __repr__(self):
        return f"<Persona(id={self.id}, nombre='{self.nombre}', apellidos='{self.apellidos}'>"


@typechecked
def main():
    log = logging.getLogger(__name__)
    log.info("Starting ..")

    # the data dir
    data_dir = os.path.join("..", "..", "data")

    # the database file
    db_file = os.path.join(data_dir, "database.sqlite")

    # creating the sqlalchemy engine
    engine = create_engine(f"sqlite:///{db_file}")

    # create the database
    log.debug("Creating the database ..")
    Base.metadata.create_all(engine)
    log.debug(".. database created.")

    # the Personas
    personas = [
        Persona(nombre="Alexis", apellidos="Chuga"),
        Persona(nombre="Benjamin", apellidos="Donoso"),
        Persona(nombre="Cesar", apellidos="Muñoz"),
    ]

    # the session maker
    session_maker = sessionmaker(bind=engine)

    # insert Personas into the dabase
    with session_maker() as session:
        session.add_all(personas)
        session.commit()

    # retrieve all the Personas from database
    with session_maker() as session:
        personas = session.query(Persona).all()
        log.debug(f"Retrieved {len(personas)} Persona from database.")

    log.info("Done.")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()
