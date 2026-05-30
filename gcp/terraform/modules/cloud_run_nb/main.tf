variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "env" {
  type = string
}

variable "registry_url" {
  type = string
}

variable "sa_email" {
  type = string
}

resource "google_cloud_run_v2_service" "nb" {
  name     = "scene-graph-nb"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.registry_url}/scene-graph-nb:latest"
      ports {
        container_port = 9500
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
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
  name     = google_cloud_run_v2_service.nb.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "url" {
  value = google_cloud_run_v2_service.nb.uri
}
