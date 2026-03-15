docker build -t azureks.azurecr.io/demos/treasury-assignment .
docker push azureks.azurecr.io/demos/treasury-assignment
helm upgrade --install treasury-assignment ../k8s/treasury-assignment -n worktracker
kubectl rollout restart deployment/treasury-assignment -n worktracker