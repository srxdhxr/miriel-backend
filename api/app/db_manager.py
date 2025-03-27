from sqlmodel import SQLModel, Session, create_engine, select
from .database import engine, DATABASE_URL
from .models import Domain, ScrapedDocument
import typer
import rich
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

def init_db():
    SQLModel.metadata.create_all(engine)

@app.command()
def create_tables():
    """Create all database tables"""
    try:
        init_db()
        console.print("[green]✓ Tables created successfully!")
    except Exception as e:
        console.print(f"[red]Error creating tables: {e}")

@app.command()
def show_tables():
    """Show all tables and their structure"""
    inspector = engine.dialect.inspector
    
    for table_name in inspector.get_table_names():
        table = Table(title=f"Table: {table_name}")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Nullable", style="yellow")
        table.add_column("Default", style="green")
        
        for column in inspector.get_columns(table_name):
            table.add_row(
                column["name"],
                str(column["type"]),
                str(column["nullable"]),
                str(column.get("default", ""))
            )
        
        console.print(table)
        console.print("\n")

@app.command()
def check_connection():
    """Test database connection"""
    try:
        with Session(engine) as session:
            session.exec(select(1)).first()
        console.print("[green]✓ Database connection successful!")
    except Exception as e:
        console.print(f"[red]Database connection failed: {e}")

if __name__ == "__main__":
    app() 