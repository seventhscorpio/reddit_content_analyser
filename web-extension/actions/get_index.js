/**
 * @file
 * Get all threads summaries from a current subreddit index page
 */

;(() => {
    const entries = document.querySelectorAll(
        '#siteTable > .thing:not(.promoted)',
    )
    const summaries = []

    for (const entry of entries) {
        const titleEl = entry.querySelector('a.title')
        const flairEl = entry.querySelector('span.linkflairlabel')

        const url = new URL(location.href)
        url.pathname = entry.dataset.permalink

        summaries.push({
            title: titleEl.textContent,
            flair: flairEl?.textContent,
            author: entry.dataset.author,
            published: +entry.dataset.timestamp,
            url: url.toString(),
        })
    }

    return summaries
})()
