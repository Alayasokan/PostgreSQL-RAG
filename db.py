from os import getenv
from pgvector.peewee import VectorField
from peewee import PostgresqlDatabase, Model, TextField, ForeignKeyField, IntegerField, CharField, DateTimeField
from datetime import datetime

db = PostgresqlDatabase(
    getenv("POSTGRES_DB_NAME"),
    host=getenv("POSTGRES_DB_HOST"),
    port=getenv("POSTGRES_DB_PORT"),
    user=getenv("POSTGRES_DB_USER"),
    password=getenv("POSTGRES_DB_PASSWORD"),
)

class Documents(Model):
    name = TextField()
    class Meta:
        database = db
        db_table = 'documents'

class Tags(Model):
    name = TextField()
    class Meta:
        database = db
        db_table = 'tags'

class DocumentTags(Model):
    document_id = ForeignKeyField(Documents, backref="document_tags", on_delete='CASCADE')
    tag_id = ForeignKeyField(Tags, backref="document_tags", on_delete='CASCADE')
    class Meta:
        database = db
        db_table = 'document_tags'

class DocumentInformationChunks(Model):
    document_id = ForeignKeyField(Documents, backref="document_information_chunks", on_delete='CASCADE')
    chunk = TextField()
    embedding = VectorField(dimensions=768)
    chunk_index = IntegerField(null=True)
    chunk_type = CharField(default='text')
    created_at = DateTimeField(default=datetime.now)
    class Meta:
        database = db
        db_table = 'document_information_chunks'

class Users(Model):
    username = TextField(unique=True)
    created_at = DateTimeField(default=datetime.now)
    class Meta:
        database = db
        db_table = 'users'

class Conversations(Model):
    user_id = ForeignKeyField(Users, backref='conversations', on_delete='CASCADE')
    message = TextField()
    role = CharField(max_length=10)
    timestamp = DateTimeField(default=datetime.now)
    class Meta:
        database = db
        db_table = 'conversations'

# Connect and create tables
db.connect()
db.create_tables([Documents, Tags, DocumentTags, DocumentInformationChunks, Users, Conversations])

# Vector index (HNSW)
db.execute_sql("CREATE INDEX IF NOT EXISTS idx_embedding ON document_information_chunks USING hnsw (embedding vector_cosine_ops);")

def set_diskann_query_rescore(query_rescore: int):
    db.execute_sql("SET hnsw.ef_search = %s", (query_rescore,))

def get_or_create_user(username: str) -> Users:
    user, created = Users.get_or_create(username=username)
    return user   # fixed syntax error