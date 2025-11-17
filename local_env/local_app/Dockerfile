# Auto-generated Dockerfile for local_app
FROM icr.io/appcafe/open-liberty:full-java11-openj9-ubi

# Optional JVM tuning environment variables
ENV JVM_HEAP_MB=1024
ENV GC_POLICY=gencon

# Copy the WAR into Liberty dropins
COPY local_app.war /config/dropins/local_app.war

EXPOSE 9080

# Default CMD comes from the base Liberty image
