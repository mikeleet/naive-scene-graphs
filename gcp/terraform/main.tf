module "artifact_registry" {
  source  = "./modules/artifact_registry"
  project = var.project_id
  region  = var.region
  env     = var.env
}

module "iam" {
  source  = "./modules/iam"
  project = var.project_id
  region  = var.region
  env     = var.env
}

module "vertex_ai_training" {
  source  = "./modules/vertex_ai_training"
  project = var.project_id
  region  = var.region
  env     = var.env
}
