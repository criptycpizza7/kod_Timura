from neo4j import GraphDatabase

def open_driver():
    URI = "neo4j://94.228.122.139:7687"
    AUTH = ("neo4j", "12345678")

    return GraphDatabase.driver(URI, auth=AUTH)

def close_driver(driver):
    driver.close()

def run_query(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]