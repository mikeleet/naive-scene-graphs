variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "env" {
  type = string
}

variable "service_name" {
  type = string
}

variable "image_tag" {
  type = string
}

variable "model_type" {
  type = string
}

variable "registry_url" {
  type = string
}

variable "sa_email" {
  type = string
}

resource "google_cloud_run_v2_service" "visual" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.registry_url}/${var.image_tag}:latest"
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "1Gi"
        }
      }
      env {
        name  = "MODEL_TYPE"
        value = var.model_type
      }
    }
    service_account = var.sa_email
  }

  labels = {
    environment = var.env
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project
  location = var.region
  name     = google_cloud_run_v2_service.visual.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "url" {
  value = google_cloud_run_v2_service.visual.uri
}
