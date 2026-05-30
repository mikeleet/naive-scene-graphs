variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "env" {
  type = string
}

resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "scene-graph"
  format        = "DOCKER"
  labels = {
    environment = var.env
  }
}

output "repo_name" {
  value = google_artifact_registry_repository.repo.repository_id
}

output "url" {
  value = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.repo.repository_id}"
}
