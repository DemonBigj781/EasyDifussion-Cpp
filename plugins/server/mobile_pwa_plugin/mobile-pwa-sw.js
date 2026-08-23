/*
 * Easy Diffusion Mobile PWA service worker.
 * Caches only the application shell. Generation/API requests remain network-only.
 */

"use strict"

const CACHE_PREFIX = "easy-diffusion-mobile-pwa-"
const CACHE_NAME = `${CACHE_PREFIX}1.1.2`
const SHELL_ASSETS = [
    "/",
    "/media/css/main.css",
    "/media/css/themes.css",
    "/media/css/fontawesome-all.min.css",
    "/media/css/fonts.css",
    "/media/js/utils.js",
    "/media/js/engine.js",
    "/media/js/parameters.js",
    "/media/js/main.js",
    "/media/js/plugins.js",
    "/plugins/user/mobile-pwa.css",
    "/plugins/user/mobile-pwa.plugin.js",
    "/media/images/icon-512x512.png",
    "/mobile-pwa-icon-192.png",
    "/mobile-pwa.webmanifest"
]

self.addEventListener("install", function (event) {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(function (cache) {
                return Promise.all(
                    SHELL_ASSETS.map(function (url) {
                        return fetch(url, { cache: "reload" })
                            .then(function (response) {
                                if (response.ok) {
                                    return cache.put(url, response)
                                }
                                return undefined
                            })
                            .catch(function () {
                                return undefined
                            })
                    })
                )
            })
            .then(function () {
                return self.skipWaiting()
            })
    )
})

self.addEventListener("activate", function (event) {
    event.waitUntil(
        caches.keys()
            .then(function (keys) {
                return Promise.all(
                    keys
                        .filter(function (key) {
                            return key.indexOf(CACHE_PREFIX) === 0 && key !== CACHE_NAME
                        })
                        .map(function (key) {
                            return caches.delete(key)
                        })
                )
            })
            .then(function () {
                return self.clients.claim()
            })
    )
})

function networkFirstNavigation(request) {
    return fetch(request)
        .then(function (response) {
            if (response.ok) {
                const copy = response.clone()
                caches.open(CACHE_NAME).then(function (cache) {
                    cache.put("/", copy)
                })
            }
            return response
        })
        .catch(function () {
            return caches.match("/")
        })
}

function staleWhileRevalidate(request, cacheKey) {
    return caches.match(cacheKey).then(function (cached) {
        const networkResponse = fetch(request)
            .then(function (response) {
                if (response.ok) {
                    const copy = response.clone()
                    caches.open(CACHE_NAME).then(function (cache) {
                        cache.put(cacheKey, copy)
                    })
                }
                return response
            })
            .catch(function () {
                return cached
            })

        return cached || networkResponse
    })
}

function networkFirstAsset(request, cacheKey) {
    return fetch(request)
        .then(function (response) {
            if (response.ok) {
                const copy = response.clone()
                caches.open(CACHE_NAME).then(function (cache) {
                    cache.put(cacheKey, copy)
                })
            }
            return response
        })
        .catch(function () {
            return caches.match(cacheKey)
        })
}

self.addEventListener("fetch", function (event) {
    const request = event.request
    if (request.method !== "GET" || request.headers.has("range")) {
        return
    }

    const url = new URL(request.url)
    if (url.origin !== self.location.origin) {
        return
    }

    if (request.mode === "navigate") {
        event.respondWith(networkFirstNavigation(request))
        return
    }

    const isCoreCode = url.pathname.indexOf("/media/") === 0
        && /\.(?:css|js|woff2?|ttf)$/i.test(url.pathname)
    const isCoreImage = [
        "/media/images/favicon-16x16.png",
        "/media/images/favicon-32x32.png",
        "/media/images/icon-512x512.png"
    ].indexOf(url.pathname) !== -1
    const isPluginAsset = url.pathname === "/plugins/user/mobile-pwa.css"
        || url.pathname === "/plugins/user/mobile-pwa.plugin.js"
        || url.pathname === "/mobile-pwa-icon-192.png"
        || url.pathname === "/mobile-pwa.webmanifest"

    if (isPluginAsset) {
        // Plugin files change independently from Easy Diffusion's core assets.
        // Prefer the network so an installed PWA receives fixes on its next load.
        event.respondWith(networkFirstAsset(request, url.pathname))
        return
    }

    if (isCoreCode || isCoreImage) {
        event.respondWith(staleWhileRevalidate(request, request))
    }
})
