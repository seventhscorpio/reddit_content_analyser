/**
 * @file
 * Get info about current page
 */

;(() => {
    const bodyEl = document.querySelector('body')

    if (bodyEl.classList.contains('listing-page')) {
        /**
         *
         * @param {string} name
         * @returns {string|null}
         */
        const checkSortType = (name) => {
            if (bodyEl.classList.contains(`${name}-page`)) {
                return name
            } else {
                return null
            }
        }

        const sort =
            checkSortType('hot') ??
            checkSortType('new') ??
            checkSortType('rising') ??
            checkSortType('controversial') ??
            checkSortType('top')

        const prevEl = document.querySelector('.nextprev > .prev-button > a')
        const nextEl = document.querySelector('.nextprev > .next-button > a')

        return {
            type: 'index',
            href: location.href,
            name: location.pathname.match(/\/r\/([^\/]+)/)[1],
            sort,
            prevPageUrl: prevEl?.href || null,
            nextPageUrl: nextEl?.href || null,
        }
    } else if (bodyEl.classList.contains('comments-page')) {
        return {
            type: 'thread',
            href: location.href,
        }
    } else {
        return {
            type: 'unsupported',
            href: location.href,
        }
    }
})()
