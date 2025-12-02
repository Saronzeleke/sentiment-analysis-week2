import subprocess
import sys
import os

def check_postgresql():
    """Check if PostgreSQL is installed and running"""
    try:
        result = subprocess.run(
            ['pg_isready'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_postgresql():
    """Provide installation instructions for PostgreSQL"""
    print("PostgreSQL is not installed or running.")
    print("\nInstallation instructions:")
    print("\nFor Ubuntu/Debian:")
    print("  sudo apt update")
    print("  sudo apt install postgresql postgresql-contrib")
    print("  sudo systemctl start postgresql")
    
    print("\nFor macOS (using Homebrew):")
    print("  brew install postgresql")
    print("  brew services start postgresql")
    
    print("\nFor Windows:")
    print("  Download from: https://www.postgresql.org/download/windows/")
    print("  Use PostgreSQL Installer")
    
    print("\nAfter installation, run this script again.")
    sys.exit(1)

def create_database():
    """Create database and tables"""
    from src.database.database_connection import DatabaseConnection, create_tables, insert_banks_data
    
    print("Creating database and tables...")
    
    db = DatabaseConnection()
    
    try:
        # Test connection
        if not db.connect():
            print("Failed to connect to database. Please check your credentials.")
            return False
        
        # Create tables
        create_tables(db)
        
        # Insert bank data
        insert_banks_data(db)
        
        print("✅ Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
        
    finally:
        db.close()

def main():
    """Main setup function"""
    print("="*50)
    print("BANK REVIEWS SENTIMENT ANALYSIS - DATABASE SETUP")
    print("="*50)
    
    # Check PostgreSQL
    if not check_postgresql():
        install_postgresql()
    
    print("✅ PostgreSQL is running")
    
    # Create database
    if create_database():
        print("\nNext steps:")
        print("1. Update your .env file with database credentials")
        print("2. Run: python src/database/data_loader.py")
        print("3. Check data/processed/ for CSV files to load")
    else:
        print("\nSetup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()