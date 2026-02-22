# Hugo Tech Blog Plan

A step-by-step plan to set up a Hugo-based tech blog, organized by topic, with a custom homepage, no comments or newsletter, public GitHub repo, and easy article addition.

## Steps
1. **Initial Setup**
   - Install Hugo on Windows.
   - Create a new Hugo site (e.g., `bharat-blog`).
   - Add PaperMod theme (or similar modern theme).
   - Initialize a public GitHub repository and push the site code.

2. **Content Organization**
   - Structure `content/` with subfolders for each topic (e.g., `ai/`, `llm/`, `devops/`).
   - Each article is a markdown file in its topic folder.
   - To add a new topic, create a new folder under `content/`.

3. **Homepage Customization**
   - Override the default homepage by creating `layouts/index.html`.
   - Design homepage to include:
     - Hero section (title, tagline, featured image)
     - Topics grid/list (with icons/descriptions)
     - Recent articles grouped by topic
     - About section (short bio or site purpose)
   - Use Hugo’s `.Site.RegularPages.GroupBy "Section"` to group articles by topic.
   - Add custom CSS as needed in `assets/` or `static/`.

4. **Configuration**
   - Set theme in config file (`config.toml` or `config.yaml`).
   - Disable comments in `[params]`.
   - Remove newsletter/contact menu items.
   - Optionally, define taxonomies for tags/categories.

5. **Article Workflow**
   - To add an article: `hugo new <topic>/<article-title>.md`.
   - Fill in front matter (title, date, tags, draft: false).
   - Commit and push to GitHub; Netlify auto-deploys.

6. **Deployment & Domain**
   - Connect GitHub repo to Netlify for free hosting.
   - Set build command to `hugo`, publish directory to `public`.
   - Add custom domain from Cloudflare in Netlify, update DNS in Cloudflare.
   - Enable HTTPS in Netlify.

7. **Verification**
   - Preview site locally with `hugo server`.
   - Confirm Netlify deployment and domain connection.
   - Test adding a new article and topic.
   - Check homepage layout and topic grouping.

## Relevant files
- `content/<topic>/<article>.md` — Article markdown files by topic
- `layouts/index.html` — Custom homepage template
- `config.toml` or `config.yaml` — Site configuration
- `static/` or `assets/` — Custom CSS/images

## Verification
1. Site builds and runs locally (`hugo server`).
2. Netlify deploys site from GitHub repo.
3. Custom domain works with HTTPS.
4. Homepage displays hero, topics, recent articles, and about section.
5. New topics and articles appear automatically when added.

## Decisions
- Organize articles by topic (not date or tag)
- No newsletter, contact form, or comments
- Public GitHub repo
- Custom homepage layout (not theme default)

## Further Considerations
1. Homepage design: Provide a sample `layouts/index.html` and CSS for a modern look.
2. Article outline: Provide a markdown template for new articles.
3. Future extensibility: Easy to add new topics, sections, or features as needed.
