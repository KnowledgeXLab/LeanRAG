# MySQL Docker Setup for LeanGraph

This directory contains Docker configuration files to quickly spin up a MySQL server that's compatible with the `database_utils.py` script.

## Files Created

- `docker-compose.yml` - Docker Compose configuration for MySQL
- `Dockerfile.mysql` - Standalone Dockerfile for MySQL
- `mysql-init/01-init.sql` - MySQL initialization script
- `mysql-docker.sh` - Management script for easy container operations

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Start MySQL server
docker-compose up -d mysql

# Check if it's running
docker-compose ps

# Stop MySQL server
docker-compose down
```

### Option 2: Using the Management Script

```bash
# Make script executable (already done)
chmod +x mysql-docker.sh

# Start MySQL
./mysql-docker.sh start

# Check status
./mysql-docker.sh status

# Connect to MySQL shell
./mysql-docker.sh connect

# View logs
./mysql-docker.sh logs

# Stop MySQL
./mysql-docker.sh stop

# Restart MySQL
./mysql-docker.sh restart

# Reset (delete all data and start fresh)
./mysql-docker.sh reset
```

### Option 3: Using Docker directly

```bash
# Build custom image
docker build -f Dockerfile.mysql -t leangraph-mysql .

# Run container
docker run -d \
  --name leangraph-mysql \
  -e MYSQL_ROOT_PASSWORD=123 \
  -e MYSQL_DATABASE=leanrag_default \
  -p 4321:3306 \
  -v mysql_data:/var/lib/mysql \
  leangraph-mysql
```

## Connection Details

The MySQL server will be available with these connection parameters (matching `database_utils.py`):

- **Host**: `localhost`
- **Port**: `4321`
- **User**: `root`
- **Password**: `123`
- **Default Database**: `leanrag_default`
- **Character Set**: `utf8mb4`
- **Collation**: `utf8mb4_unicode_ci`

## Database Schema

The `database_utils.py` script will automatically create these tables:

1. **entities** - Entity information with descriptions, source IDs, degrees, parents, and levels
2. **relations** - Relationship data between entities with descriptions, weights, and levels  
3. **communities** - Community data with entity names, descriptions, and findings

## Troubleshooting

### Container won't start
```bash
# Check Docker logs
docker logs leangraph-mysql

# Or use the management script
./mysql-docker.sh logs
```

### Connection refused
```bash
# Make sure container is running
./mysql-docker.sh status

# Check if port 4321 is available
netstat -an | grep 4321

# Try connecting manually
mysql -h localhost -P 4321 -u root -p123
```

### Reset everything
```bash
# This will delete all data and start fresh
./mysql-docker.sh reset
```

### Permission issues
```bash
# Make sure the script is executable
chmod +x mysql-docker.sh
```

## Data Persistence

MySQL data is stored in a Docker volume named `mysql_data`. This means your data will persist even if you stop and restart the container. To completely reset the database, use:

```bash
./mysql-docker.sh reset
```

Or manually remove the volume:

```bash
docker volume rm mysql_data
```
