# db_connect.py



from sshtunnel import SSHTunnelForwarder
import vertica_python

# -----------------------------
# SSH CONFIG
# -----------------------------
SSH_HOST = "big-dama-3.dima.tu-berlin.de"
SSH_PORT = 22
SSH_USERNAME = "automap"
SSH_PASSWORD = "uTBcwZaocBqQa8Cpp5bE"

# -----------------------------
# VERTICA CONFIG
# -----------------------------
VERTICA_HOST = "big-dama-1.dima.tu-berlin.de"
VERTICA_PORT = 5433
VERTICA_USER = "automap"
VERTICA_PASSWORD = "automapd2ip"
VERTICA_DATABASE = "xformer"

# -----------------------------
# CREATE SSH TUNNEL
# -----------------------------
server = SSHTunnelForwarder(
    (SSH_HOST, SSH_PORT),
    ssh_username=SSH_USERNAME,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=(VERTICA_HOST, VERTICA_PORT)
)

server.start()

print(f"SSH tunnel established on local port: {server.local_bind_port}")

# -----------------------------
# CONNECT TO VERTICA
# -----------------------------
conn_info = {
    'host': '127.0.0.1',
    'port': server.local_bind_port,
    'user': VERTICA_USER,
    'password': VERTICA_PASSWORD,
    'database': VERTICA_DATABASE,
    'autocommit': True,
}

def create_connection():
    server.start()

    conn = vertica_python.connect(**conn_info)
    print("Connected to Vertica!")
    return conn, server

def test_connection(conn):
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("Vertica Version:", cur.fetchone())
    cur.close()

def close_connection(conn, server):
    conn.close()
    server.stop()
    print("Connections closed.")