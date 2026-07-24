/** @type {import('next').NextConfig} */
const nextConfig = {
  // The dashboard imports TYPES from workspace packages (@compound/api,
  // @compound/storage) for type-safety; their runtime is never bundled, but
  // transpilePackages keeps Next from choking on the TS source it may touch.
  transpilePackages: ["@compound/api", "@compound/storage"],
  reactStrictMode: true,
};

export default nextConfig;
