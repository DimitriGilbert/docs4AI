# Use Bun as base image
FROM --platform=linux/amd64 oven/bun:alpine

# Set working directory
WORKDIR /app

# Install shadow for user management
RUN apk add --no-cache shadow

# Copy package files first for better layer caching
COPY package.json ./
COPY bun.lockb* ./
COPY bun.lock* ./
COPY yarn.lock* ./
COPY package-lock.json* ./
COPY pnpm-lock.yaml* ./

# Install dependencies based on lock file
RUN if [ -f bun.lockb ] || [ -f bun.lock ]; then \
        bun install; \
    elif [ -f yarn.lock ]; then \
        yarn install; \
    elif [ -f package-lock.json ]; then \
        npm install; \
    elif [ -f pnpm-lock.yaml ]; then \
        pnpm install; \
    else \
        echo "No lockfile found! Using bun install" && \
        bun install; \
    fi

# Copy the rest of the application
COPY . .

# Create user with dynamic UID if it doesn't exist
ARG UID=1000
RUN if ! getent passwd $UID; then \
        useradd -u $UID -d /app -s /bin/sh appuser; \
    fi && \
    chown -R $UID:$UID /app

# Switch to appuser
USER $UID

# Expose port
EXPOSE 30947

# Start development server
CMD ["bun", "run", "dev"]
