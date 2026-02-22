# Implementation Tasks for Hugo Tech Blog

This checklist will help you implement the plan step by step. Mark each task as you complete it.

## 1. Initial Setup
- [ ] Install Hugo (Windows)
- [ ] Create new Hugo site (`hugo new site bharat-blog`)
- [ ] Add PaperMod theme
- [ ] Initialize Git repository
- [ ] Create public GitHub repo and push code

## 2. Content Organization
- [ ] Create topic folders under `content/` (e.g., `ai/`, `llm/`, `devops/`)
- [ ] Add a sample article in each topic folder
- [ ] Test adding a new topic folder

## 3. Homepage Customization
- [ ] Create/override `layouts/index.html` for custom homepage
- [ ] Design homepage: hero, topics grid, recent articles, about section
- [ ] Add custom CSS (if needed)

## 4. Configuration
- [ ] Set theme in `config.toml` or `config.yaml`
- [ ] Disable comments in `[params]`
- [ ] Remove newsletter/contact menu items
- [ ] (Optional) Define taxonomies for tags/categories

## 5. Article Workflow
- [ ] Add a new article using `hugo new <topic>/<article-title>.md`
- [ ] Fill in front matter (title, date, tags, draft: false)
- [ ] Commit and push to GitHub
- [ ] Confirm Netlify auto-deploys changes

## 6. Deployment & Domain
- [ ] Connect GitHub repo to Netlify
- [ ] Set build command to `hugo`, publish directory to `public`
- [ ] Add custom domain from Cloudflare in Netlify
- [ ] Update DNS in Cloudflare
- [ ] Enable HTTPS in Netlify

## 7. Verification
- [ ] Preview site locally (`hugo server`)
- [ ] Confirm Netlify deployment and domain connection
- [ ] Test adding a new article and topic
- [ ] Check homepage layout and topic grouping

---

## Notes
- Use this checklist to track your progress.
- You can add more tasks as needed for future features or improvements.
