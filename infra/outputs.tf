output "s3_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "rds_endpoint" {
  value     = aws_db_instance.postgres.endpoint
  sensitive = true
}

output "rds_database_url" {
  value     = "postgresql+asyncpg://flora:${var.db_password}@${aws_db_instance.postgres.endpoint}/flora"
  sensitive = true
}
