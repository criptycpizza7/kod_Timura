from neo4j import GraphDatabase

URI = "bolt://localhost"
AUTH = ("neo4j", "12345678")

def open_driver():
    URI = "bolt://localhost"
    AUTH = ("neo4j", "12345678")

    return GraphDatabase.driver(URI, auth=AUTH)

def close_driver(driver):
    driver.close()

def run_query(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]