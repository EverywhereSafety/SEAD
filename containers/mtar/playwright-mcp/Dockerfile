FROM node@sha256:7725a5c2c83eed1d36258c66efae14b1ceccd021db9ed1d9559d3335ed3d68ed

WORKDIR /opt/sead/playwright-mcp
COPY package.json package-lock.json ./
RUN npm ci --omit=dev \
    && npx playwright install --with-deps chrome \
    && npm cache clean --force

ENV MCP_PLAYWRIGHT_PORT=9092
EXPOSE 9092
CMD ["sh", "-c", "exec ./node_modules/.bin/playwright-mcp --host 0.0.0.0 --port \"$MCP_PLAYWRIGHT_PORT\" --allowed-hosts '*' --headless --isolated --no-sandbox --shared-browser-context --output-dir /workspace/.playwright-mcp"]
