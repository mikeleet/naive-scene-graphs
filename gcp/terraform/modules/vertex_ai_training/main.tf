variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "env" {
  type = string
}

resource "google_storage_bucket" "data" {
  name     = "${var.project}-scene-graph"
  location = var.region
  labels = {
    environment = var.env
  }
}

output "bucket_name" {
  value = google_storage_bucket.data.name
}
