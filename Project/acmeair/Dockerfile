# Start from a full image with all features installed
FROM icr.io/appcafe/open-liberty:full-java17-openj9-ubi
COPY server.xml /config/server.xml
COPY acmeair.war /config/dropins/
EXPOSE 9080