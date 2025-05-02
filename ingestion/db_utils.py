from sqlalchemy import create_engine, inspect

def get_engine(user, password, host, port, db):
    return create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}', future=True)

def table_exists(engine, table_name):
    inspector = inspect(engine)
    return inspector.has_table(table_name)
