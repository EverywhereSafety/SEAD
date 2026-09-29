FROM node@sha256:7725a5c2c83eed1d36258c66efae14b1ceccd021db9ed1d9559d3335ed3d68ed

# This image is built and published before evaluation.  Runtime workers never
# invoke npm/npx or mutate this software layer.
RUN npm install --global --omit=dev \
    supergateway@3.4.3 \
    @modelcontextprotocol/server-filesystem@2026.7.10 \
    && npm cache clean --force

USER node
