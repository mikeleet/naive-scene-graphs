variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "credentials_file" {
  type    = string
  default = "~/gcp-key.json"
}

variable "env" {
  type    = string
  default = "demo"
}
