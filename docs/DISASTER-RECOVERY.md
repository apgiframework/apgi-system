# Disaster Recovery Procedures

## Overview

This document outlines the disaster recovery procedures for the APGI system. Disaster recovery encompasses procedures for recovering from catastrophic failures, data loss, system-wide outages, and other major incidents that threaten system availability and data integrity.

## Disaster Classification

### Level 1: Minor Incident

- Single component failure
- Temporary service degradation
- Data inconsistency in non-critical components

**Recovery Time Objective (RTO)**: < 1 hour
**Recovery Point Objective (RPO)**: < 15 minutes

### Level 2: Major Incident

- Multiple component failures
- Partial system outage
- Significant data loss in secondary systems

**Recovery Time Objective (RTO)**: < 4 hours
**Recovery Point Objective (RPO)**: < 1 hour

### Level 3: Critical Disaster

- Complete system failure
- Total data center outage
- Widespread data corruption

**Recovery Time Objective (RTO)**: < 24 hours
**Recovery Point Objective (RPO)**: < 4 hours

### Level 4: Catastrophic Disaster

- Permanent loss of primary data center
- Complete destruction of infrastructure
- Loss of critical personnel

**Recovery Time Objective (RTO)**: < 72 hours
**Recovery Point Objective (RPO)**: < 24 hours

## Recovery Preparation

### Backup Strategy

#### Primary Backup Systems

1. **Database Backups**
   - Full backups: Daily at 02:00 UTC
   - Incremental backups: Every 4 hours
   - Transaction log backups: Every 15 minutes
   - Retention: 30 days for incrementals, 1 year for full backups

2. **Configuration Backups**
   - System configuration: Every 6 hours
   - User configurations: Real-time with changes
   - Environment configurations: Daily

3. **Code and Artifact Backups**
   - Source code: Continuous via Git
   - Docker images: Registry with immutable tags
   - Dependencies: Cached and version-pinned

#### Backup Storage Locations

```yaml
backup_locations:
  primary:
    type: cloud_storage
    provider: aws_s3
    bucket: apgi-production-backups
    region: us-west-2
    encryption: aes256

  secondary:
    type: cloud_storage
    provider: gcp_cloud_storage
    bucket: apgi-backup-secondary
    region: us-central1
    encryption: aes256

  tertiary:
    type: on_premises
    location: secure_facility
    encryption: hardware_security_module
```

### Recovery Environment

#### Primary Recovery Site

- **Location**: AWS us-west-2 (Oregon)
- **Capacity**: 100% of production capacity
- **Activation Time**: < 2 hours
- **Data Synchronization**: Real-time replication

#### Secondary Recovery Site

- **Location**: GCP us-central1 (Iowa)
- **Capacity**: 50% of production capacity
- **Activation Time**: < 6 hours
- **Data Synchronization**: Hourly replication

#### Tertiary Recovery Site

- **Location**: On-premises secure facility
- **Capacity**: 25% of production capacity
- **Activation Time**: < 24 hours
- **Data Synchronization**: Daily replication

## Recovery Procedures

### Automated Recovery Scripts

#### System Health Check Script

```bash
#!/bin/bash
# system_health_check.sh

echo "=== APGI System Health Check ==="
echo "Timestamp: $(date -u)"

# Check core services
services=("apgi-api" "apgi-worker" "redis" "postgresql" "nginx")
failed_services=()

for service in "${services[@]}"; do
    if ! systemctl is-active --quiet "$service"; then
        echo "❌ Service $service is not running"
        failed_services+=("$service")
    else
        echo "✅ Service $service is running"
    fi
done

# Check database connectivity
if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo "❌ Database is not accessible"
    failed_services+=("database")
else
    echo "✅ Database is accessible"
fi

# Check Redis connectivity
if ! redis-cli ping >/dev/null 2>&1; then
    echo "❌ Redis is not accessible"
    failed_services+=("redis")
else
    echo "✅ Redis is accessible"
fi

# Check disk space
disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$disk_usage" -gt 90 ]; then
    echo "❌ Disk usage is ${disk_usage}% (>90%)"
    failed_services+=("disk_space")
else
    echo "✅ Disk usage is ${disk_usage}%"
fi

# Check memory usage
memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$memory_usage" -gt 95 ]; then
    echo "❌ Memory usage is ${memory_usage}% (>95%)"
    failed_services+=("memory")
else
    echo "✅ Memory usage is ${memory_usage}%"
fi

if [ ${#failed_services[@]} -eq 0 ]; then
    echo "🎉 All systems operational"
    exit 0
else
    echo "⚠️  Issues detected with: ${failed_services[*]}"
    exit 1
fi
```

#### Automated Recovery Script

```bash
#!/bin/bash
# automated_recovery.sh

echo "=== APGI Automated Recovery ==="
echo "Started at: $(date -u)"

# Load environment variables
source /etc/apgi/environment

# Step 1: Stop all services gracefully
echo "Stopping services..."
for service in "${SERVICES[@]}"; do
    systemctl stop "$service"
    sleep 5
done

# Step 2: Verify database integrity
echo "Checking database integrity..."
if ! pg_checksums --check "$PGDATA"; then
    echo "Database integrity check failed, attempting repair..."
    # Attempt database repair
    systemctl stop postgresql
    pg_resetwal "$PGDATA"
    systemctl start postgresql
fi

# Step 3: Restore from latest backup if needed
if [ "$DATABASE_CORRUPTION_DETECTED" = "true" ]; then
    echo "Restoring database from backup..."
    LATEST_BACKUP=$(find /backups/database -name "*.sql.gz" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$LATEST_BACKUP" ]; then
        gunzip < "$LATEST_BACKUP" | psql -U apgi -d apgi_production
        echo "Database restored from $LATEST_BACKUP"
    else
        echo "ERROR: No database backup found!"
        exit 1
    fi
fi

# Step 4: Restart services in dependency order
echo "Restarting services..."
declare -a service_order=("postgresql" "redis" "apgi-worker" "apgi-api" "nginx")

for service in "${service_order[@]}"; do
    echo "Starting $service..."
    systemctl start "$service"

    # Wait for service to be healthy
    timeout=60
    while [ $timeout -gt 0 ]; do
        if systemctl is-active --quiet "$service"; then
            echo "✅ $service started successfully"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done

    if [ $timeout -le 0 ]; then
        echo "❌ Failed to start $service"
        exit 1
    fi
done

# Step 5: Run health checks
echo "Running health checks..."
if ./system_health_check.sh; then
    echo "🎉 Recovery completed successfully"
    exit 0
else
    echo "❌ Health checks failed after recovery"
    exit 1
fi
```

### Manual Recovery Procedures

#### Procedure 1: Service Restart Recovery

**Trigger**: Individual service failures

**Steps**:

1. Identify failed service using monitoring dashboard
2. Check service logs for error details
3. Attempt graceful service restart:

   ```bash
   sudo systemctl restart <service_name>
   ```

4. If restart fails, check dependencies and restart them first
5. Verify service health and connectivity
6. Update incident ticket with resolution

#### Procedure 2: Database Recovery

**Trigger**: Database connectivity issues or corruption

**Steps**:

1. Check database server status:

   ```bash
   sudo systemctl status postgresql
   ```

2. If server is down, restart it:

   ```bash
   sudo systemctl restart postgresql
   ```

3. Check database integrity:

   ```bash
   sudo -u postgres pg_checksums --check /var/lib/postgresql/data
   ```

4. If corruption detected, restore from backup:

   ```bash
   # Stop dependent services
   sudo systemctl stop apgi-api apgi-worker

   # Restore from backup
   sudo -u postgres psql -d apgi_production < /backups/database/latest.sql

   # Restart services
   sudo systemctl start apgi-worker apgi-api
   ```

5. Verify data integrity and application functionality

#### Procedure 3: Full System Recovery

**Trigger**: Complete system failure

**Steps**:

1. Assess damage and determine recovery scope
2. Activate backup infrastructure (AWS/GCP)
3. Restore system from latest backups:
   - Deploy infrastructure using Terraform/IaC
   - Restore database from S3/GCS backups
   - Deploy application containers
   - Restore configuration files
4. Verify system functionality with comprehensive tests
5. Redirect traffic to recovered system
6. Decommission failed infrastructure

#### Procedure 4: Data Loss Recovery

**Trigger**: Significant data loss or corruption

**Steps**:

1. Stop all write operations to prevent further corruption
2. Identify extent of data loss using backup verification
3. Restore data from appropriate backup level:
   - Point-in-time recovery for recent losses
   - Full backup restoration for widespread corruption
4. Verify data consistency and referential integrity
5. Implement additional safeguards if needed
6. Resume normal operations with monitoring

### Cloud-Specific Recovery

#### AWS Recovery Procedures

```bash
# Deploy recovery infrastructure
cd infrastructure/aws
terraform workspace select recovery
terraform apply -auto-approve

# Restore RDS database
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier apgi-recovery \
    --db-snapshot-identifier latest-production-snapshot \
    --db-instance-class db.r5.large

# Deploy application
kubectl apply -f k8s/recovery/

# Update DNS to point to recovery environment
aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch file://dns-recovery.json
```

#### GCP Recovery Procedures

```bash
# Create recovery instance
gcloud compute instances create apgi-recovery \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud

# Restore from GCS backup
gsutil -m cp -r gs://apgi-backup-secondary/latest/* /data/

# Deploy containers
docker-compose -f docker-compose.recovery.yml up -d

# Update load balancer
gcloud compute backend-services add-backend apgi-backend \
    --instance-group=apgi-recovery-group \
    --instance-group-zone=us-central1-a
```

## Communication Plan

### Internal Communication

1. **Immediate Response Team**: Slack channel #incident-response
2. **Engineering Team**: Email distribution list <engineering@apgi-system.com>
3. **Management**: SMS and email alerts to on-call managers
4. **Documentation**: Internal wiki updates with recovery progress

### External Communication

1. **Customer Notification**: Status page updates and email notifications
2. **Stakeholder Updates**: Regular progress reports during extended outages
3. **Public Relations**: Press releases for significant incidents

### Communication Templates

#### Customer Notification Template

```text
Subject: APGI System Service Interruption - Recovery Update

Dear APGI Customer,

We are currently experiencing a service interruption affecting [specific services].
Our team is actively working to restore service.

Current Status: [Recovery in Progress / Service Restored]
Estimated Resolution: [Time estimate or "Resolved"]
Impact: [Description of impact]

We apologize for any inconvenience this may cause. For real-time updates,
please visit our status page at https://status.apgi-system.com.

Best regards,
APGI Operations Team
```

## Testing and Validation

### Recovery Testing Schedule

- **Weekly**: Component-level recovery testing
- **Monthly**: Full system recovery simulation
- **Quarterly**: Cross-region failover testing
- **Annually**: Complete disaster recovery exercise

### Recovery Validation Checklist

```markdown
## Recovery Validation Checklist

### Infrastructure
- [ ] All servers/services started
- [ ] Network connectivity established
- [ ] Load balancers configured correctly
- [ ] DNS records updated

### Application
- [ ] API endpoints responding
- [ ] Database connections working
- [ ] Background jobs processing
- [ ] User authentication functioning

### Data
- [ ] Database integrity verified
- [ ] Data consistency checks passed
- [ ] Backup verification completed
- [ ] Replication synchronized

### Monitoring
- [ ] Alerting systems active
- [ ] Logging collecting data
- [ ] Metrics being reported
- [ ] Dashboard accessible

### Security
- [ ] Access controls restored
- [ ] Encryption keys loaded
- [ ] Security policies applied
- [ ] Compliance requirements met

### User Acceptance
- [ ] Sample user workflows tested
- [ ] Performance benchmarks met
- [ ] Feature functionality verified
- [ ] User feedback collected
```

## Performance Metrics

### Recovery Time Objectives (RTO)

| Component | RTO Target | Current Performance | Last Tested |
| ---------- | ----------- | -------------------- | ------------ |
| API Service | 15 minutes | 8 minutes | 2024-01-15 |
| Database | 1 hour | 32 minutes | 2024-01-15 |
| Full System | 4 hours | 2.5 hours | 2024-01-10 |
| Cross-Region | 8 hours | 5 hours | 2024-01-05 |

### Recovery Point Objectives (RPO)

| Data Type | RPO Target | Current Performance | Last Tested |
| ---------- | ----------- | -------------------- | ------------ |
| User Data | 15 minutes | 5 minutes | 2024-01-15 |
| System Config | 1 hour | 30 minutes | 2024-01-15 |
| Analytics | 4 hours | 1 hour | 2024-01-15 |
| Archives | 24 hours | 6 hours | 2024-01-15 |

## Continuous Improvement

### Post-Incident Review Process

1. **Timeline Reconstruction**: Document all events and responses
2. **Root Cause Analysis**: Identify contributing factors and root causes
3. **Impact Assessment**: Evaluate business and technical impacts
4. **Lessons Learned**: Document insights and improvement opportunities
5. **Action Items**: Create specific, measurable improvement tasks

### Improvement Tracking

```yaml
improvement_projects:
  - id: backup_performance
    title: "Improve backup completion time"
    target: "Reduce full backup time by 50%"
    deadline: "2024-03-01"
    status: "in_progress"

  - id: monitoring_coverage
    title: "Expand monitoring coverage"
    target: "Add monitoring for 95% of system components"
    deadline: "2024-02-15"
    status: "planned"

  - id: automation_testing
    title: "Automate recovery testing"
    target: "Implement automated weekly recovery tests"
    deadline: "2024-04-01"
    status: "planned"
```

## Contact Information

### Emergency Contacts

- **Primary On-Call**: +1-555-0101 (24/7)
- **Secondary On-Call**: +1-555-0102 (24/7)
- **Infrastructure Lead**: +1-555-0103
- **Database Administrator**: +1-555-0104
- **Security Officer**: +1-555-0105

### External Resources

- **AWS Support**: 1-888-280-4331 (Enterprise Support)
- **GCP Support**: 1-855-836-5473 (Premium Support)
- **CloudFlare**: Emergency line for DNS issues
- **Backup Vendor**: 24/7 support line

## Revision History

| Version | Date | Author | Changes |
| -------- | ----- | ------- | -------- |
| 1.0 | 2024-01-01 | Operations Team | Initial document |
| 1.1 | 2024-01-15 | SRE Team | Added automated recovery scripts |
| 1.2 | 2024-01-20 | Security Team | Enhanced security procedures |
| 2.0 | 2024-02-01 | Operations Team | Major revision with cloud procedures |
