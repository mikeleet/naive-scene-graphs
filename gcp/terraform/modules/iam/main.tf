variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "env" {
  type = string
}

resource "google_service_account" "sa" {
  account_id   = "scene-graph-deployer"
  display_name = "scene graph deployer"
}

resource "google_project_iam_member" "run_invoker" {
  project = var.project
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.sa.email}"
}

resource "google_project_iam_member" "artifact_reader" {
  project = var.project
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.sa.email}"
}

output "sa_email" {
  value = google_service_account.sa.email
}
