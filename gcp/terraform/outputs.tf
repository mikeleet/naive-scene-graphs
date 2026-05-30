output "artifact_registry_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${module.artifact_registry.repo_name}"
}

output "gcs_bucket" {
  value = module.vertex_ai_training.bucket_name
}

# cloud run outputs — uncomment after images are pushed
# output "nb_service_url" {
#   value = module.cloud_run_nb.url
# }
# output "visual_pretrained_url" {
#   value = module.cloud_run_visual_pretrained.url
# }
# output "visual_trained_url" {
#   value = module.cloud_run_visual_trained.url
# }
