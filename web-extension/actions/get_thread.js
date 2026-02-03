/**
 * @file
 * Get full thread
 */

;(() => {
    const entry = document.querySelector('#siteTable .thing')
    const titleEl = entry?.querySelector('a.title')
    const flairEl = entry?.querySelector('span.linkflairlabel')
    const thread = {
        title: titleEl?.textContent,
        text: entry?.querySelector('.usertext-body')?.textContent,
        flair: flairEl?.textContent,
        author: entry?.dataset.author,
        published: +entry?.dataset.timestamp,
        url: entry?.dataset.permalink,
        comments: [],
    }

    /**
     *
     * @param {HTMLElement} containerEl
     * @param {{comments: Object[]}}}
     */
    function getAllComments(containerEl, node) {
        for (const commentEl of containerEl.children) {
            if (!commentEl.classList.contains('comment')) {
                continue
            }

            const contentEl = commentEl.querySelector('.entry')
            const childrenEl = commentEl.querySelector('.child > .listing')

            const comment = {
                author: contentEl.querySelector('.author')?.textContent,
                text: contentEl.querySelector('.usertext-body')?.textContent,
                published: contentEl
                    .querySelector('time')
                    ?.getAttribute('datetime'),
                score: {
                    likes: +contentEl
                        .querySelector('.score.likes')
                        ?.getAttribute('title'),
                    dislikes: +contentEl
                        .querySelector('.score.dislikes')
                        ?.getAttribute('title'),
                    unvoted: +contentEl
                        .querySelector('.score.unvoted')
                        ?.getAttribute('title'),
                },
                comments: [],
            }

            node.comments.push(comment)

            if (childrenEl) {
                getAllComments(childrenEl, comment)
            }
        }
    }

    // Get all nested comments
    getAllComments(
        document.querySelector('.commentarea > .nestedlisting'),
        thread,
    )

    return thread
})()
